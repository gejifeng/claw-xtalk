// -----------------------------------------------------------------------
// SessionRegistry – manages the three-way ID mapping
// browserSessionId 1:1 xtalkSessionId, N:1 openclawSessionKey
// -----------------------------------------------------------------------
import { v4 as uuidv4 } from "uuid";
import { SessionMapping, TurnContext, TurnState } from "../types/state";

export class SessionRegistry {
  private sessions = new Map<string, SessionMapping>();

  /** Register a new browser connection. Returns the created mapping. */
  register(
    browserSessionId: string,
    openclawSessionKey = `xtalk:${browserSessionId}`,
  ): SessionMapping {
    const xtalkSessionId = `x-${uuidv4()}`;
    const mapping: SessionMapping = {
      browserSessionId,
      xtalkSessionId,
      openclawSessionKey,
      currentTurn: null,
      connectedAt: Date.now(),
    };
    this.sessions.set(browserSessionId, mapping);
    return mapping;
  }

  get(browserSessionId: string): SessionMapping | undefined {
    return this.sessions.get(browserSessionId);
  }

  getByXtalkSessionId(xtalkSessionId: string): SessionMapping | undefined {
    for (const m of this.sessions.values()) {
      if (m.xtalkSessionId === xtalkSessionId) return m;
    }
    return undefined;
  }

  remove(browserSessionId: string): void {
    this.sessions.delete(browserSessionId);
  }

  /** Create a fresh turn on the given session and return it. */
  newTurn(browserSessionId: string): TurnContext {
    const mapping = this.sessions.get(browserSessionId);
    if (!mapping) throw new Error(`No session for browserSessionId=${browserSessionId}`);
    const turn: TurnContext = {
      turnId: `t-${uuidv4()}`,
      state: "Idle",
      partialTranscript: "",
      finalTranscript: "",
      agentRunId: null,
      assistantText: "",
      pendingTTSBuffer: "",
      firstChunkSent: false,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    mapping.currentTurn = turn;
    return turn;
  }

  setTurnState(browserSessionId: string, state: TurnState): void {
    const turn = this.sessions.get(browserSessionId)?.currentTurn;
    if (turn) {
      turn.state = state;
      turn.updatedAt = Date.now();
    }
  }

  all(): SessionMapping[] {
    return [...this.sessions.values()];
  }
}
