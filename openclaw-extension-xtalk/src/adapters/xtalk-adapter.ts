// -----------------------------------------------------------------------
// XtalkAdapter – WebSocket client connecting the Extension to the X-Talk
// sidecar service. Translates sidecar events into internal adapter events.
// -----------------------------------------------------------------------
import WebSocket from "ws";
import { EventEmitter } from "events";
import { SidecarConnectionState } from "../types/state";
import type { AsrTiming } from "../types/protocol";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 16000;
const MAX_RECONNECT_ATTEMPTS = 10;

export type XtalkAdapterEvent =
  | { event: "connected" }
  | { event: "disconnected" }
  | { event: "asr.partial"; sessionId: string; turnId: string; text: string }
  | { event: "asr.final"; sessionId: string; turnId: string; text: string; timing?: AsrTiming }
  | { event: "barge_in"; sessionId: string; turnId: string }
  | { event: "tts.audio"; sessionId: string; turnId: string; audioBase64: string; mimeType: string; sampleRate: number; seq: number }
  | { event: "playback.started"; sessionId: string; turnId: string }
  | { event: "playback.finished"; sessionId: string; turnId: string };

export class XtalkAdapter extends EventEmitter {
  private ws: WebSocket | null = null;
  private state: SidecarConnectionState = "disconnected";
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly sidecarUrl: string) {
    super();
  }

  connect(): void {
    if (this.state === "connecting" || this.state === "connected") return;
    this._doConnect();
  }

  disconnect(): void {
    this.reconnectAttempts = MAX_RECONNECT_ATTEMPTS;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }

  private _doConnect(): void {
    this.state = "connecting";
    console.log(`[XtalkAdapter] connecting to ${this.sidecarUrl}`);
    const ws = new WebSocket(this.sidecarUrl);
    this.ws = ws;

    ws.on("open", () => {
      this.state = "connected";
      this.reconnectAttempts = 0;
      console.log("[XtalkAdapter] connected");
      this.emit("adapter", { event: "connected" });
    });

    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        this._handleSidecarMessage(msg);
      } catch {
        // Binary audio data from sidecar – Phase 1B
      }
    });

    ws.on("close", () => {
      this.state = "disconnected";
      console.warn("[XtalkAdapter] disconnected");
      this.emit("adapter", { event: "disconnected" });
      this._scheduleReconnect();
    });

    ws.on("error", (err) => {
      console.error("[XtalkAdapter] ws error", err.message);
      this.state = "error";
    });
  }

  private _scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) return;
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.reconnectAttempts, RECONNECT_MAX_MS);
    this.reconnectAttempts++;
    console.log(`[XtalkAdapter] reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    this.reconnectTimer = setTimeout(() => this._doConnect(), delay);
  }

  private _handleSidecarMessage(msg: Record<string, unknown>): void {
    switch (msg["type"]) {
      case "session.opened":
        console.log(`[XtalkAdapter] session opened: ${msg["sessionId"]}`);
        break;
      case "asr.partial":
        this.emit("adapter", {
          event: "asr.partial",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
          text: msg["text"],
        });
        break;
      case "asr.final":
        this.emit("adapter", {
          event: "asr.final",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
          text: msg["text"],
          timing: msg["timing"] as AsrTiming | undefined,
        });
        break;
      case "barge_in":
        this.emit("adapter", {
          event: "barge_in",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
        });
        break;
      case "tts.audio":
        this.emit("adapter", {
          event: "tts.audio",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
          audioBase64: msg["audioBase64"],
          mimeType: msg["mimeType"],
          sampleRate: msg["sampleRate"],
          seq: msg["seq"],
        });
        break;
      case "playback.started":
        this.emit("adapter", {
          event: "playback.started",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
        });
        break;
      case "playback.finished":
        this.emit("adapter", {
          event: "playback.finished",
          sessionId: msg["sessionId"],
          turnId: msg["turnId"],
        });
        break;
      case "error":
        console.error("[XtalkAdapter] sidecar error", msg["message"]);
        break;
    }
  }

  openSession(sessionId: string, turnId?: string): void {
    this._send({ type: "session.open", sessionId, ...(turnId ? { turnId } : {}) });
  }

  closeSession(sessionId: string): void {
    this._send({ type: "session.close", sessionId });
  }

  sendAudioFrame(sessionId: string, seq: number, pcm: Buffer): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const header = JSON.stringify({ type: "audio.frame", sessionId, seq });
    this.ws.send(header);
    this.ws.send(pcm);
  }

  finishAudio(sessionId: string): void {
    this._send({ type: "audio.stop", sessionId });
  }

  ttsSendChunk(sessionId: string, turnId: string, text: string): void {
    this._send({ type: "tts.enqueue", sessionId, turnId, text });
  }

  ttsFlush(sessionId: string, turnId: string): void {
    this._send({ type: "tts.flush", sessionId, turnId });
  }

  stopPlayback(sessionId: string, turnId: string): void {
    this._send({ type: "playback.stop", sessionId, turnId });
  }

  /** Notify the sidecar that a new turn has started (resets ASR state). */
  notifyNewTurn(sessionId: string, turnId: string): void {
    this.openSession(sessionId, turnId);
  }

  private _send(msg: object): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn("[XtalkAdapter] dropping message – not connected", (msg as { type: string }).type);
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  get connectionState(): SidecarConnectionState {
    return this.state;
  }
}
