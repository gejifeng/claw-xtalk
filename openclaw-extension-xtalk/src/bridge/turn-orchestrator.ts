// -----------------------------------------------------------------------
// TurnOrchestrator – drives the Turn state machine and coordinates the
// flow: ASR final -> OpenClaw agent -> TTS chunks -> playback complete.
// -----------------------------------------------------------------------
import { EventEmitter } from "events";
import { SessionRegistry } from "./session-registry";
import { ChunkingPolicy } from "./chunking-policy";
import { XtalkAdapter } from "../adapters/xtalk-adapter";
import { OpenclawAgentAdapter } from "../adapters/openclaw-agent-adapter";
import type { AsrTiming } from "../types/protocol";

const ASR_FILLER_RE = /^(?:嗯+|啊+|哦+|呃+|额+|唉+|哎+|诶+|欸+|噢+|哼+|嗯哼+|啊哈+)$/;

function normalizeAsrText(text: string): string {
  return text
    .replace(/[\s\p{P}\p{S}]+/gu, "")
    .trim()
    .toLowerCase();
}

function isIgnorableUtterance(text: string): boolean {
  const normalized = normalizeAsrText(text);
  if (!normalized) return true;
  return ASR_FILLER_RE.test(normalized);
}

export class TurnOrchestrator extends EventEmitter {
  // Tracks the active 150 ms forced-flush timer per browser session.
  private readonly _firstChunkTimers = new Map<string, ReturnType<typeof setTimeout>>();

  constructor(
    private readonly registry: SessionRegistry,
    private readonly xtalk: XtalkAdapter,
    private readonly agent: OpenclawAgentAdapter,
    private readonly chunking: ChunkingPolicy,
  ) {
    super();
  }

  onAsrPartial(browserSessionId: string, turnId: string, text: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn || mapping.currentTurn.turnId !== turnId) return;
    mapping.currentTurn.partialTranscript = text;
    this.registry.setTurnState(browserSessionId, "Transcribing");
    this.emit("event", { kind: "ASR_PARTIAL", browserSessionId, turnId, text });
  }

  async onAsrFinal(
    browserSessionId: string,
    turnId: string,
    text: string,
    timing?: AsrTiming,
  ): Promise<void> {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn || mapping.currentTurn.turnId !== turnId) return;
    const trimmed = text.trim();
    if (isIgnorableUtterance(trimmed)) {
      console.log(`[TurnOrchestrator] ignoring filler/noise transcript bid=${browserSessionId} text=${JSON.stringify(trimmed)}`);
      mapping.currentTurn.partialTranscript = "";
      mapping.currentTurn.finalTranscript = "";
      this.registry.setTurnState(browserSessionId, "Listening");
      this.emit("event", {
        kind: "ASR_IGNORED",
        browserSessionId,
        turnId,
        text: trimmed,
        reason: "filler",
      });
      return;
    }

    mapping.currentTurn.finalTranscript = trimmed;
    this.registry.setTurnState(browserSessionId, "Thinking");
    this.emit("event", { kind: "ASR_FINAL", browserSessionId, turnId, text: trimmed, timing });

    console.log(`[TurnOrchestrator] starting agent run bid=${browserSessionId} sessionKey=${mapping.openclawSessionKey} text="${trimmed.slice(0, 60)}"`);

    try {
      this.emit("event", { kind: "AGENT_RUN_STARTED", browserSessionId, turnId });
      await this.agent.runStream(
        mapping.openclawSessionKey,
        trimmed,
        (delta) => this.onAgentDelta(browserSessionId, turnId, delta),
        (final) => this.onAgentFinal(browserSessionId, turnId, final),
      );
      console.log(`[TurnOrchestrator] agent run completed bid=${browserSessionId}`);
    } catch (err) {
      console.error("[TurnOrchestrator] agent run error", err);
      this.registry.setTurnState(browserSessionId, "Idle");
    }
  }

  private onAgentDelta(browserSessionId: string, turnId: string, delta: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn || mapping.currentTurn.turnId !== turnId) return;

    const isFirstDelta = mapping.currentTurn.state === "Thinking";
    if (isFirstDelta) {
      console.log(`[TurnOrchestrator] first delta received bid=${browserSessionId}`);
      this.registry.setTurnState(browserSessionId, "Speaking");
      // Start a 150 ms deadline: if no sentence boundary arrives in time, force
      // dispatch whatever is buffered so TTS can start without waiting for
      // punctuation.  The timer is cancelled the moment the first chunk is sent.
      const timer = setTimeout(() => {
        this._firstChunkTimers.delete(browserSessionId);
        const m = this.registry.get(browserSessionId);
        if (!m?.currentTurn || m.currentTurn.turnId !== turnId) return;
        if (m.currentTurn.firstChunkSent) return;
        const buffered = m.currentTurn.pendingTTSBuffer;
        if (buffered.trim().length === 0) return;
        m.currentTurn.pendingTTSBuffer = "";
        m.currentTurn.firstChunkSent = true;
        console.log(`[TurnOrchestrator] 150ms deadline flush bid=${browserSessionId} chars=${buffered.length}`);
        this.xtalk.ttsSendChunk(m.xtalkSessionId, turnId, buffered);
      }, 150);
      this._firstChunkTimers.set(browserSessionId, timer);
    }
    this.emit("event", { kind: "AGENT_DELTA", browserSessionId, turnId, text: delta });

    mapping.currentTurn.assistantText += delta;
    mapping.currentTurn.pendingTTSBuffer += delta;

    // skipMinLen=true until the first chunk has been sent so that a short
    // opening clause is dispatched immediately on the first sentence boundary.
    const skipMinLen = !mapping.currentTurn.firstChunkSent;
    const chunks = this.chunking.extractReady(mapping.currentTurn.pendingTTSBuffer, skipMinLen);
    if (chunks.ready.length > 0) {
      mapping.currentTurn.pendingTTSBuffer = chunks.remaining;
      for (const chunk of chunks.ready) {
        if (!mapping.currentTurn.firstChunkSent) {
          mapping.currentTurn.firstChunkSent = true;
          // Cancel the deadline timer – a boundary was found in time.
          const timer = this._firstChunkTimers.get(browserSessionId);
          if (timer !== undefined) {
            clearTimeout(timer);
            this._firstChunkTimers.delete(browserSessionId);
          }
        }
        this.xtalk.ttsSendChunk(mapping.xtalkSessionId, turnId, chunk);
      }
    }
  }

  private onAgentFinal(browserSessionId: string, turnId: string, final: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn || mapping.currentTurn.turnId !== turnId) return;
    const assistantText = mapping.currentTurn.assistantText;
    const unsentSuffix = final.startsWith(assistantText)
      ? final.slice(assistantText.length)
      : final;
    const remaining = mapping.currentTurn.pendingTTSBuffer + unsentSuffix;
    mapping.currentTurn.assistantText = final;
    mapping.currentTurn.pendingTTSBuffer = "";
    if (remaining.trim().length > 0) {
      this.xtalk.ttsSendChunk(mapping.xtalkSessionId, turnId, remaining);
    }
    this.xtalk.ttsFlush(mapping.xtalkSessionId, turnId);
    this.emit("event", { kind: "AGENT_FINAL", browserSessionId, turnId, text: final });
  }

  onBargeIn(browserSessionId: string, turnId: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn) return;
    this.emit("event", {
      kind: "BARGE_IN_DETECTED",
      browserSessionId,
      turnId: mapping.currentTurn.turnId,
    });
    this.interruptCurrentTurn(browserSessionId);
  }

  interruptCurrentTurn(browserSessionId: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn) return;
    const { turnId } = mapping.currentTurn;
    const { xtalkSessionId } = mapping;

    // Cancel any pending first-chunk deadline timer before the turn is replaced.
    const pendingTimer = this._firstChunkTimers.get(browserSessionId);
    if (pendingTimer !== undefined) {
      clearTimeout(pendingTimer);
      this._firstChunkTimers.delete(browserSessionId);
    }

    this.registry.setTurnState(browserSessionId, "Interrupted");
    this.xtalk.stopPlayback(xtalkSessionId, turnId);
    if (mapping.currentTurn.agentRunId) {
      this.agent.cancelRun(mapping.openclawSessionKey, mapping.currentTurn.agentRunId);
    }
    this.emit("event", { kind: "RUN_CANCELLED", browserSessionId, turnId });

    const newTurn = this.registry.newTurn(browserSessionId);
    this.registry.setTurnState(browserSessionId, "Listening");
    this.xtalk.notifyNewTurn(mapping.xtalkSessionId, newTurn.turnId);
  }

  onPlaybackFinished(browserSessionId: string, turnId: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn || mapping.currentTurn.turnId !== turnId) return;
    this.registry.setTurnState(browserSessionId, "Idle");
    this.emit("event", { kind: "TURN_COMPLETED", browserSessionId, turnId });

    const newTurn = this.registry.newTurn(browserSessionId);
    this.registry.setTurnState(browserSessionId, "Listening");
    this.xtalk.notifyNewTurn(mapping.xtalkSessionId, newTurn.turnId);
  }
}
