// -----------------------------------------------------------------------
// Web routes – mounts the Browser Voice UI and the Browser↔Extension WS.
// -----------------------------------------------------------------------
import path from "path";
import type { Application } from "express";
import { WebSocketServer, WebSocket } from "ws";
import { SessionRegistry } from "../bridge/session-registry";
import { TurnOrchestrator } from "../bridge/turn-orchestrator";
import { InterruptController } from "../bridge/interrupt-controller";
import { XtalkAdapter } from "../adapters/xtalk-adapter";
import type { InternalEvent } from "../types/protocol";

// __dirname is dist/web/ at runtime; UI assets are copied there by the build step
const UI_DIR = path.join(__dirname, "ui");

export function mountRoutes(
  app: Application,
  wss: WebSocketServer,
  registry: SessionRegistry,
  orchestrator: TurnOrchestrator,
  interruptCtrl: InterruptController,
  xtalk: XtalkAdapter,
): void {
  // ---- Static UI --------------------------------------------------------
  app.use("/ui", (_req, res) => {
    res.sendFile(path.join(UI_DIR, "index.html"));
  });

  // ---- Health check -------------------------------------------------------
  app.get("/health", (_req, res) => {
    res.json({
      status: "ok",
      sidecarState: xtalk.connectionState,
      sessions: registry.all().length,
    });
  });

  // ---- Browser WebSocket ------------------------------------------------
  wss.on("connection", (ws: WebSocket, req) => {
    console.log(`[routes] browser connected from ${req.socket.remoteAddress}`);
    let browserSessionId: string | null = null;

    const onEvent = (ev: InternalEvent) => {
      if (ev.browserSessionId !== browserSessionId) return;
      const msg = internalEventToExtMsg(ev);
      if (msg) safeSend(ws, msg);
    };
    orchestrator.on("event", onEvent);

    ws.on("message", (raw) => {
      // Binary -> PCM audio frame
      if (raw instanceof Buffer && browserSessionId) {
        const mapping = registry.get(browserSessionId);
        if (!mapping) return;
        const seq = Date.now();
        xtalk.sendAudioFrame(mapping.xtalkSessionId, seq, raw);
        return;
      }

      try {
        const msg = JSON.parse(raw.toString());
        handleBrowserMsg(msg);
      } catch {
        console.warn("[routes] unparseable browser message");
      }
    });

    ws.on("close", () => {
      orchestrator.off("event", onEvent);
      if (browserSessionId) {
        const mapping = registry.get(browserSessionId);
        if (mapping) xtalk.closeSession(mapping.xtalkSessionId);
        registry.remove(browserSessionId);
        console.log(`[routes] browser session removed sid=${browserSessionId}`);
      }
    });

    function handleBrowserMsg(msg: Record<string, unknown>): void {
      switch (msg["type"]) {
        case "session.init": {
          browserSessionId = msg["browserSessionId"] as string;
          const mapping = registry.register(browserSessionId);
          const turn = registry.newTurn(browserSessionId);
          if (xtalk.connectionState === "connected") {
            xtalk.openSession(mapping.xtalkSessionId, turn.turnId);
          } else {
            console.warn(
              `[routes] sidecar not connected yet; will defer session.open xid=${mapping.xtalkSessionId}`,
            );
          }
          registry.setTurnState(browserSessionId, "Idle");
          safeSend(ws, {
            type: "session.ready",
            xtalkSessionId: mapping.xtalkSessionId,
            openclawSessionKey: mapping.openclawSessionKey,
          });
          console.log(
            `[routes] session init bid=${browserSessionId} xid=${mapping.xtalkSessionId} turn=${turn.turnId}`,
          );
          break;
        }
        case "audio.start":
          if (browserSessionId) {
            registry.setTurnState(browserSessionId, "Listening");
            const mapping = registry.get(browserSessionId);
            if (mapping?.currentTurn && xtalk.connectionState === "connected") {
              xtalk.openSession(mapping.xtalkSessionId, mapping.currentTurn.turnId);
            }
          }
          break;
        case "audio.stop":
          if (browserSessionId) {
            registry.setTurnState(browserSessionId, "Idle");
            const mapping = registry.get(browserSessionId);
            if (mapping) xtalk.finishAudio(mapping.xtalkSessionId);
          }
          break;
        case "playback.stop":
          if (browserSessionId) interruptCtrl.handleUserRequested(browserSessionId);
          break;
        case "conversation.reset":
          if (browserSessionId) interruptCtrl.handleUserRequested(browserSessionId);
          break;
      }
    }
  });
}

// ---- Map internal events to Browser protocol messages -------------------
function internalEventToExtMsg(ev: InternalEvent): object | null {
  switch (ev.kind) {
    case "ASR_PARTIAL":     return { type: "asr.partial", text: ev.text };
    case "ASR_FINAL":       return { type: "asr.final", text: ev.text, timing: ev.timing };
    case "ASR_IGNORED":     return { type: "asr.ignored", text: ev.text, reason: ev.reason };
    case "AGENT_DELTA":     return { type: "assistant.delta", text: ev.text };
    case "AGENT_FINAL":     return { type: "assistant.final", text: ev.text };
    case "TTS_AUDIO_CHUNK": return {
      type: "tts.audio",
      audioBase64: ev.audioBase64,
      mimeType: ev.mimeType,
      sampleRate: ev.sampleRate,
      seq: ev.seq,
    };
    case "BARGE_IN_DETECTED": return { type: "interrupt.detected" };
    case "TTS_PLAYBACK_STARTED": return { type: "playback.state", state: "speaking" };
    case "TURN_COMPLETED":  return { type: "playback.state", state: "idle" };
    default:                return null;
  }
}

function safeSend(ws: WebSocket, msg: object): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}
