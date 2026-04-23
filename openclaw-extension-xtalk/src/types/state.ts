// -----------------------------------------------------------------------
// state.ts – Turn and session state machine types.
// -----------------------------------------------------------------------

export type TurnState =
  | "Idle"
  | "Listening"
  | "Transcribing"
  | "Thinking"
  | "Speaking"
  | "Interrupted";

export interface TurnContext {
  turnId: string;
  state: TurnState;
  partialTranscript: string;
  finalTranscript: string;
  agentRunId: string | null;
  /** Full assistant text accumulated from streaming deltas */
  assistantText: string;
  /** Accumulates assistant text waiting to be chunked and sent to TTS */
  pendingTTSBuffer: string;
  /** True once the first TTS chunk has been dispatched for this turn */
  firstChunkSent: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface SessionMapping {
  browserSessionId: string;
  xtalkSessionId: string;
  openclawSessionKey: string;
  currentTurn: TurnContext | null;
  connectedAt: number;
}

export type SidecarConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

export interface SidecarState {
  connectionState: SidecarConnectionState;
  url: string;
  reconnectAttempts: number;
}
