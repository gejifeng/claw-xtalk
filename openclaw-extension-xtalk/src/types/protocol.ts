// -----------------------------------------------------------------------
// protocol.ts – All message types crossing the three transport boundaries:
//   1. Browser ↔ Extension WebSocket  (BrowserToExtMsg / ExtToBrowserMsg)
//   2. Extension ↔ X-Talk Sidecar WS  (ExtToSidecarMsg / SidecarToExtMsg)
//   3. Internal event bus              (InternalEvent)
// -----------------------------------------------------------------------

// ── Browser → Extension ─────────────────────────────────────────────────
export type BrowserToExtMsg =
  | { type: "session.init"; browserSessionId: string }
  | { type: "audio.start" }
  | { type: "audio.stop" }
  | { type: "audio.mute"; muted: boolean }
  | { type: "conversation.reset" }
  | { type: "playback.stop" };

export interface AsrTiming {
  speechEndedAtMs: number;
  sttLatencyMs: number;
  endpointWaitMs: number;
  transcribeDurationMs: number;
}

// ── Extension → Browser ─────────────────────────────────────────────────
export type ExtToBrowserMsg =
  | { type: "session.ready"; xtalkSessionId: string; openclawSessionKey: string }
  | { type: "asr.partial"; text: string }
  | { type: "asr.final"; text: string; timing?: AsrTiming }
  | { type: "asr.ignored"; text: string; reason: string }
  | { type: "assistant.delta"; text: string }
  | { type: "assistant.final"; text: string }
  | { type: "tts.audio"; audioBase64: string; mimeType: string; sampleRate: number; seq: number }
  | { type: "playback.state"; state: PlaybackState }
  | { type: "interrupt.detected" }
  | { type: "error"; code: ErrorCode; message: string };

export type PlaybackState = "idle" | "speaking" | "interrupted";

export type ErrorCode =
  | "XTALK_UNAVAILABLE"
  | "XTALK_ASR_FAILED"
  | "AGENT_TIMEOUT"
  | "AGENT_DISCONNECTED"
  | "TTS_FAILED"
  | "PLAYBACK_FAILED"
  | "SESSION_MAPPING_LOST";

// ── Extension → Sidecar ─────────────────────────────────────────────────
export type ExtToSidecarMsg =
  | { type: "session.open"; sessionId: string; turnId?: string }
  | { type: "session.close"; sessionId: string }
  | { type: "audio.frame"; seq: number; sessionId: string }
  | { type: "audio.stop"; sessionId: string }
  | { type: "tts.enqueue"; sessionId: string; turnId: string; text: string }
  | { type: "tts.flush"; sessionId: string; turnId: string }
  | { type: "playback.stop"; sessionId: string; turnId: string };

// ── Sidecar → Extension ─────────────────────────────────────────────────
export type SidecarToExtMsg =
  | { type: "session.opened"; sessionId: string }
  | { type: "asr.partial"; sessionId: string; turnId: string; text: string }
  | { type: "asr.final"; sessionId: string; turnId: string; text: string; timing?: AsrTiming }
  | { type: "barge_in"; sessionId: string; turnId: string }
  | { type: "tts.audio"; sessionId: string; turnId: string; audioBase64: string; mimeType: string; sampleRate: number; seq: number }
  | { type: "playback.started"; sessionId: string; turnId: string }
  | { type: "playback.finished"; sessionId: string; turnId: string }
  | { type: "error"; turnId?: string; message: string };

// ── Internal event bus ───────────────────────────────────────────────────
export type InternalEvent =
  | { kind: "USER_AUDIO_STARTED"; browserSessionId: string }
  | { kind: "ASR_PARTIAL"; browserSessionId: string; turnId: string; text: string }
  | { kind: "ASR_FINAL"; browserSessionId: string; turnId: string; text: string; timing?: AsrTiming }
  | { kind: "ASR_IGNORED"; browserSessionId: string; turnId: string; text: string; reason: string }
  | { kind: "AGENT_RUN_STARTED"; browserSessionId: string; turnId: string }
  | { kind: "AGENT_DELTA"; browserSessionId: string; turnId: string; text: string }
  | { kind: "AGENT_FINAL"; browserSessionId: string; turnId: string; text: string }
  | { kind: "BARGE_IN_DETECTED"; browserSessionId: string; turnId: string }
  | { kind: "RUN_CANCELLED"; browserSessionId: string; turnId: string }
  | { kind: "TTS_AUDIO_CHUNK"; browserSessionId: string; turnId: string; audioBase64: string; mimeType: string; sampleRate: number; seq: number }
  | { kind: "TTS_PLAYBACK_STARTED"; browserSessionId: string; turnId: string }
  | { kind: "TURN_COMPLETED"; browserSessionId: string; turnId: string };
