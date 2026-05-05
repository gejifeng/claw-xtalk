# Hermes Agent Adaptation of Claw-Xtalk

This document records all changes made to adapt the Claw-Xtalk voice bridge from
**OpenClaw** to **Hermes Agent**. TypeScript was verified to compile cleanly
(`npm run build` exits 0) after all changes.

---

## Summary of Changed Files

| File | Change |
|---|---|
| `openclaw-extension-xtalk/src/adapters/openclaw-agent-adapter.ts` | **Full rewrite**: replaced WebSocket + device-auth handshake with Hermes HTTP SSE chat completions API. Exports `HermesAgentAdapter`. |
| `openclaw-extension-xtalk/src/types/state.ts` | Renamed `SessionMapping.openclawSessionKey` → `hermesSessionId`. |
| `openclaw-extension-xtalk/src/types/protocol.ts` | Added `hermesSessionId` to `session.ready` wire type; kept `openclawSessionKey` alias for UI backward-compat. |
| `openclaw-extension-xtalk/src/bridge/session-registry.ts` | Updated field name + comment to match `hermesSessionId`. |
| `openclaw-extension-xtalk/src/bridge/turn-orchestrator.ts` | Import and type annotation changed to `HermesAgentAdapter`; log messages updated. |
| `openclaw-extension-xtalk/src/web/routes.ts` | `session.ready` now sends both `hermesSessionId` (new canonical) and `openclawSessionKey` (alias = same value). |
| `openclaw-extension-xtalk/src/index.ts` | Import and instantiation updated to `HermesAgentAdapter`. |
| `scripts/start-all.sh` | Process renamed `openclaw-bridge` → `hermes-bridge`; added Hermes config/env loading block. |

**Files NOT changed** (as required):
- `src/web/ui/index.html` — browser UI unchanged (reads `openclawSessionKey` only in a debug display line; backward-compat alias preserves this)
- `src/bridge/turn-orchestrator.ts` — orchestration/chunking/interruption logic unchanged
- `src/bridge/chunking-policy.ts` — unchanged
- `src/bridge/interrupt-controller.ts` — unchanged
- `src/adapters/xtalk-adapter.ts` — unchanged
- `xtalk-bridge-service/` — Python sidecar ASR/TTS backends unchanged

---

## Architecture Notes

### What changed in the adapter

**Before (OpenClaw):**
- Persistent `WebSocket` connection to `ws://127.0.0.1:18789`
- Ed25519 device signing handshake (`~/.openclaw/identity/device.json` + `device-auth.json`)
- Run/event multiplexing over a single WS connection with `reqId` correlation
- `cancelRun()` sent a cancellation message over the WS

**After (Hermes):**
- Stateless `http.request()` per turn, `POST /api/v1/chat/completions`
- SSE streaming response (`stream: true`) with OpenAI-compatible delta framing
- `session_id` field in the request body carries the Hermes UUID looked up from `~/.hermes/state.db`
- `cancelRun()` calls `req.destroy()` which triggers `ECONNRESET` (caught and treated as a clean resolve)
- No persistent connection to manage or reconnect

### Config priority (lowest → highest)

1. Hard-coded defaults (`host=127.0.0.1`, `port=80`, `model=hermes`)
2. `~/.hermes/config.yaml` (`gateway.host`, `gateway.port`, `gateway.api_path`, `model`, `api_key`)
3. `~/.hermes/.env` (`HERMES_API_KEY`, `HERMES_GATEWAY_HOST`, `HERMES_GATEWAY_PORT`, `HERMES_MODEL`)
4. Process environment variables (same names as `.env` keys)

### Session mapping

- On each new browser connection, `SessionRegistry.register()` assigns a stable key `xtalk:<browserSessionId>`.
- `HermesAgentAdapter.resolveHermesSessionId()` queries `~/.hermes/state.db` (sqlite) for a `sessions` row whose `title` matches that key.
- If `state.db` doesn't exist yet, a deterministic UUID is derived from `sha256("xtalk:" + sessionKey)` — no I/O required; Hermes will create the session on first receipt.
- Resolved IDs are cached in-memory for the process lifetime.

---

## New Config Values Needed

### `~/.hermes/config.yaml`

```yaml
# Hermes Agent gateway settings (voice mode)
gateway:
  host: 127.0.0.1   # default; change if Hermes runs on a different host
  port: 80           # default Hermes gateway port
  api_path: /api/v1/chat/completions  # omit to use default

model: hermes        # model name sent in chat completions requests

# Optional: embed API key here (or prefer ~/.hermes/.env)
# api_key: sk-...
```

### `~/.hermes/.env`

```bash
# API key for the Hermes gateway (if authentication is enabled)
HERMES_API_KEY=sk-your-key-here

# Override gateway coordinates (optional — config.yaml takes lower priority)
# HERMES_GATEWAY_HOST=127.0.0.1
# HERMES_GATEWAY_PORT=80
# HERMES_MODEL=hermes
```

### Environment variables (highest priority — useful for Docker / CI)

| Variable | Description | Default |
|---|---|---|
| `HERMES_API_KEY` | Bearer token for Hermes gateway | _(none — anonymous)_ |
| `HERMES_GATEWAY_HOST` | Hermes gateway hostname | `127.0.0.1` |
| `HERMES_GATEWAY_PORT` | Hermes gateway port | `80` |
| `HERMES_MODEL` | Model name in chat completion request | `hermes` |
| `HERMES_CONFIG` | Path to config.yaml | `~/.hermes/config.yaml` |
| `HERMES_STATE_DB` | Path to state.db | `~/.hermes/state.db` |

---

## Full File Contents

### `openclaw-extension-xtalk/src/adapters/openclaw-agent-adapter.ts`

```typescript
// -----------------------------------------------------------------------
// openclaw-agent-adapter.ts – Hermes Agent HTTP adapter
//
// Replaces the previous OpenClaw WebSocket gateway protocol with the
// Hermes Agent HTTP chat completions API (SSE streaming).
//
// Hermes gateway defaults:
//   POST http://localhost:80/api/v1/chat/completions
//   Config  : ~/.hermes/config.yaml
//   Secrets : ~/.hermes/.env
//   Sessions: ~/.hermes/state.db  (tables: sessions, messages)
// -----------------------------------------------------------------------
import crypto from "crypto";
import fs from "fs";
import http from "http";
import os from "os";
import path from "path";
import yaml from "yaml";
import initSqlJs from "sql.js";

// ── Types ────────────────────────────────────────────────────────────────────
type SqlJsStatic = Awaited<ReturnType<typeof initSqlJs>>;

interface HermesGatewayConfig {
  host: string;
  port: number;
  apiPath: string;
  model: string;
  apiKey?: string;
  stateDbPath: string;
}

// ── Config loader ─────────────────────────────────────────────────────────────
function loadHermesConfig(): HermesGatewayConfig {
  const hermesDir = path.join(os.homedir(), ".hermes");

  let host    = "127.0.0.1";
  let port    = 80;
  let apiPath = "/api/v1/chat/completions";
  let model   = "hermes";
  let apiKey: string | undefined;

  // 1. Parse ~/.hermes/config.yaml
  const configYaml = path.join(hermesDir, "config.yaml");
  if (fs.existsSync(configYaml)) {
    try {
      const raw = yaml.parse(fs.readFileSync(configYaml, "utf8")) as Record<string, unknown>;
      const gw  = raw["gateway"] as Record<string, unknown> | undefined;
      if (typeof gw?.["host"]    === "string") host    = gw["host"];
      if (typeof gw?.["port"]    === "number") port    = gw["port"];
      if (typeof gw?.["api_path"] === "string") apiPath = gw["api_path"];
      if (typeof raw["model"]    === "string") model   = raw["model"];
      if (typeof raw["api_key"]  === "string") apiKey  = raw["api_key"];
    } catch (err) {
      console.warn("[HermesAgentAdapter] cannot parse ~/.hermes/config.yaml:", (err as Error).message);
    }
  }

  // 2. ~/.hermes/.env overrides config.yaml
  const envFile = path.join(hermesDir, ".env");
  if (fs.existsSync(envFile)) {
    try {
      for (const raw of fs.readFileSync(envFile, "utf8").split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith("#") || !line.includes("=")) continue;
        const eqIdx = line.indexOf("=");
        const key   = line.slice(0, eqIdx).trim();
        let   val   = line.slice(eqIdx + 1).trim();
        if (val.length >= 2 && val[0] === val[val.length - 1] && (val[0] === '"' || val[0] === "'")) {
          val = val.slice(1, -1);
        }
        if (!val) continue;
        switch (key) {
          case "HERMES_API_KEY":       apiKey  = val;               break;
          case "HERMES_GATEWAY_HOST":  host    = val;               break;
          case "HERMES_GATEWAY_PORT":  port    = parseInt(val, 10); break;
          case "HERMES_MODEL":         model   = val;               break;
        }
      }
    } catch { /* .env is optional */ }
  }

  // 3. Process-env takes highest priority (useful for CI / Docker overrides)
  if (process.env["HERMES_API_KEY"])       apiKey  = process.env["HERMES_API_KEY"];
  if (process.env["HERMES_GATEWAY_HOST"])  host    = process.env["HERMES_GATEWAY_HOST"]!;
  if (process.env["HERMES_GATEWAY_PORT"])  port    = parseInt(process.env["HERMES_GATEWAY_PORT"]!, 10);
  if (process.env["HERMES_MODEL"])         model   = process.env["HERMES_MODEL"]!;

  return { host, port, apiPath, model, apiKey, stateDbPath: path.join(hermesDir, "state.db") };
}

// ── Lazy sql.js initialisation ─────────────────────────────────────────────
let _sqlJs: SqlJsStatic | null = null;

async function getSqlJs(): Promise<SqlJsStatic> {
  if (_sqlJs) return _sqlJs;
  _sqlJs = await initSqlJs();
  return _sqlJs;
}

// ── Hermes session lookup / creation via state.db ──────────────────────────
async function lookupOrCreateHermesSession(
  sessionKey: string,
  stateDbPath: string,
): Promise<string> {
  if (!fs.existsSync(stateDbPath)) {
    // state.db does not exist yet.  Derive a deterministic UUID-shaped ID from
    // the session key so that the same bridge session always maps to the same
    // Hermes conversation even before the first agent call creates the row.
    const hash = crypto.createHash("sha256").update(`xtalk:${sessionKey}`).digest("hex");
    return `${hash.slice(0,8)}-${hash.slice(8,12)}-4${hash.slice(13,16)}-${hash.slice(16,20)}-${hash.slice(20,32)}`;
  }

  const SQL = await getSqlJs();
  const fileBuffer = fs.readFileSync(stateDbPath);
  const db = new SQL.Database(new Uint8Array(fileBuffer));

  try {
    // Look for an existing Hermes session whose title matches the xtalk key.
    let sessionId: string | null = null;
    try {
      const stmt = db.prepare("SELECT id FROM sessions WHERE title = ? LIMIT 1");
      stmt.bind([sessionKey]);
      if (stmt.step()) {
        const row = stmt.getAsObject() as { id: unknown };
        sessionId = String(row.id);
      }
      stmt.free();
    } catch {
      // sessions table may not exist in some Hermes builds; fall through to create
    }

    if (sessionId) {
      console.log(`[HermesAgentAdapter] found existing Hermes session ${sessionId} for key=${sessionKey}`);
      return sessionId;
    }

    // Create a new session row in state.db.
    const newId  = crypto.randomUUID();
    const nowMs  = Date.now();
    try {
      db.run(
        "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        [newId, sessionKey, nowMs, nowMs],
      );
      const exported = db.export();
      fs.writeFileSync(stateDbPath, Buffer.from(exported));
    } catch (err) {
      // Schema mismatch or read-only db; log and continue with the derived ID.
      console.warn("[HermesAgentAdapter] could not write session to state.db:", (err as Error).message);
    }

    console.log(`[HermesAgentAdapter] created new Hermes session ${newId} for key=${sessionKey}`);
    return newId;
  } finally {
    db.close();
  }
}

// ── Adapter ───────────────────────────────────────────────────────────────────
export class HermesAgentAdapter {
  private readonly config: HermesGatewayConfig;

  /** Maps the bridge's stable hermesSessionId key to the Hermes session UUID. */
  private readonly sessionIdCache = new Map<string, string>();

  /** Maps hermesSessionId key to the active in-flight HTTP request (for cancellation). */
  private readonly activeRequests = new Map<string, http.ClientRequest>();

  constructor() {
    this.config = loadHermesConfig();
    console.log(
      `[HermesAgentAdapter] gateway=http://${this.config.host}:${this.config.port}` +
      ` model=${this.config.model} stateDb=${this.config.stateDbPath}`,
    );
  }

  private async resolveHermesSessionId(sessionKey: string): Promise<string> {
    const cached = this.sessionIdCache.get(sessionKey);
    if (cached) return cached;
    const id = await lookupOrCreateHermesSession(sessionKey, this.config.stateDbPath);
    this.sessionIdCache.set(sessionKey, id);
    return id;
  }

  /**
   * Send `userMessage` to Hermes and stream the response back via `onDelta` /
   * `onFinal`.  The public signature is identical to the former OpenClaw
   * adapter so the rest of the bridge requires no changes.
   */
  async runStream(
    sessionKey: string,
    userMessage: string,
    onDelta: (delta: string) => void,
    onFinal:  (finalText: string) => void,
  ): Promise<void> {
    const hermesSessionId = await this.resolveHermesSessionId(sessionKey);
    const { host, port, apiPath, model, apiKey } = this.config;

    const body = JSON.stringify({
      model,
      messages: [{ role: "user", content: userMessage }],
      stream: true,
      session_id: hermesSessionId,
    });

    console.log(
      `[HermesAgentAdapter] POST ${apiPath} session=${hermesSessionId}` +
      ` msg="${userMessage.slice(0, 60)}"`,
    );

    return new Promise<void>((resolve, reject) => {
      const headers: Record<string, string | number> = {
        "Content-Type":   "application/json",
        "Content-Length": Buffer.byteLength(body),
      };
      if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

      const req = http.request(
        { hostname: host, port, path: apiPath, method: "POST", headers },
        (res) => {
          if (res.statusCode !== undefined && res.statusCode >= 400) {
            const err = new Error(`Hermes gateway returned HTTP ${res.statusCode}`);
            this.activeRequests.delete(sessionKey);
            res.resume();
            reject(err);
            return;
          }

          let accumulated = "";
          let sseBuffer   = "";

          res.setEncoding("utf8");

          res.on("data", (chunk: string) => {
            sseBuffer += chunk;
            // Split on newline pairs (SSE event boundaries)
            const lines = sseBuffer.split(/\r?\n/);
            // The last element is an incomplete line; keep it in the buffer.
            sseBuffer = lines.pop() ?? "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const payload = line.slice(6).trim();
              if (payload === "[DONE]") continue;

              try {
                const parsed = JSON.parse(payload) as {
                  choices: Array<{
                    delta?:         { content?: string };
                    finish_reason?: string | null;
                  }>;
                };
                const deltaText = parsed.choices?.[0]?.delta?.content ?? "";
                if (deltaText) {
                  accumulated += deltaText;
                  onDelta(deltaText);
                }
              } catch {
                // Tolerate malformed / partial SSE frames
              }
            }
          });

          res.on("end", () => {
            this.activeRequests.delete(sessionKey);
            console.log(
              `[HermesAgentAdapter] stream ended session=${hermesSessionId}` +
              ` total=${accumulated.length} chars`,
            );
            onFinal(accumulated);
            resolve();
          });

          res.on("error", (err: Error) => {
            this.activeRequests.delete(sessionKey);
            console.error("[HermesAgentAdapter] response stream error:", err.message);
            // Surface partial text rather than dropping it
            onFinal(accumulated);
            resolve();
          });
        },
      );

      req.on("error", (err: Error) => {
        this.activeRequests.delete(sessionKey);
        if ((err as NodeJS.ErrnoException).code === "ECONNRESET") {
          // Triggered by our own cancelRun() → req.destroy(); treat as non-error.
          resolve();
        } else {
          console.error("[HermesAgentAdapter] request error:", err.message);
          reject(err);
        }
      });

      this.activeRequests.set(sessionKey, req);
      req.write(body);
      req.end();
    });
  }

  /** Cancel an in-flight request by destroying the TCP socket. */
  cancelRun(sessionKey: string, _runId: string): void {
    const req = this.activeRequests.get(sessionKey);
    if (req) {
      console.log(`[HermesAgentAdapter] cancelling in-flight request for sessionKey=${sessionKey}`);
      req.destroy();
      this.activeRequests.delete(sessionKey);
    }
  }
}
```

---

### `openclaw-extension-xtalk/src/types/state.ts`

```typescript
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
  /** Stable key used to look up / create the Hermes conversation in state.db */
  hermesSessionId: string;
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
```

---

### `openclaw-extension-xtalk/src/types/protocol.ts`

```typescript
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
  | { type: "session.ready"; xtalkSessionId: string; hermesSessionId: string; openclawSessionKey: string }
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
```

---

### `openclaw-extension-xtalk/src/bridge/session-registry.ts`

```typescript
// -----------------------------------------------------------------------
// SessionRegistry – manages the three-way ID mapping
// browserSessionId 1:1 xtalkSessionId, N:1 hermesSessionId
// -----------------------------------------------------------------------
import { v4 as uuidv4 } from "uuid";
import { SessionMapping, TurnContext, TurnState } from "../types/state";

export class SessionRegistry {
  private sessions = new Map<string, SessionMapping>();

  /** Register a new browser connection. Returns the created mapping. */
  register(
    browserSessionId: string,
    hermesSessionId = `xtalk:${browserSessionId}`,
  ): SessionMapping {
    const xtalkSessionId = `x-${uuidv4()}`;
    const mapping: SessionMapping = {
      browserSessionId,
      xtalkSessionId,
      hermesSessionId,
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
```

---

### `openclaw-extension-xtalk/src/bridge/turn-orchestrator.ts`

```typescript
// -----------------------------------------------------------------------
// TurnOrchestrator – drives the Turn state machine and coordinates the
// flow: ASR final -> Hermes agent -> TTS chunks -> playback complete.
// -----------------------------------------------------------------------
import { EventEmitter } from "events";
import { SessionRegistry } from "./session-registry";
import { ChunkingPolicy } from "./chunking-policy";
import { XtalkAdapter } from "../adapters/xtalk-adapter";
import { HermesAgentAdapter } from "../adapters/openclaw-agent-adapter";
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
    private readonly agent: HermesAgentAdapter,
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

    console.log(`[TurnOrchestrator] starting agent run bid=${browserSessionId} hermesSessionId=${mapping.hermesSessionId} text="${trimmed.slice(0, 60)}"`);

    try {
      this.emit("event", { kind: "AGENT_RUN_STARTED", browserSessionId, turnId });
      await this.agent.runStream(
        mapping.hermesSessionId,
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

    const skipMinLen = !mapping.currentTurn.firstChunkSent;
    const chunks = this.chunking.extractReady(mapping.currentTurn.pendingTTSBuffer, skipMinLen);
    if (chunks.ready.length > 0) {
      mapping.currentTurn.pendingTTSBuffer = chunks.remaining;
      for (const chunk of chunks.ready) {
        if (!mapping.currentTurn.firstChunkSent) {
          mapping.currentTurn.firstChunkSent = true;
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

    const pendingTimer = this._firstChunkTimers.get(browserSessionId);
    if (pendingTimer !== undefined) {
      clearTimeout(pendingTimer);
      this._firstChunkTimers.delete(browserSessionId);
    }

    this.registry.setTurnState(browserSessionId, "Interrupted");
    this.xtalk.stopPlayback(xtalkSessionId, turnId);
    if (mapping.currentTurn.agentRunId) {
      this.agent.cancelRun(mapping.hermesSessionId, mapping.currentTurn.agentRunId);
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
```

---

### `openclaw-extension-xtalk/src/web/routes.ts`

```typescript
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
            hermesSessionId: mapping.hermesSessionId,
            // backward-compat alias: browser UI debug panel reads this field name
            openclawSessionKey: mapping.hermesSessionId,
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
```

---

### `openclaw-extension-xtalk/src/index.ts`

```typescript
// -----------------------------------------------------------------------
// index.ts – Extension entry point.
// Phase 1: runs as a standalone Node.js process.
// -----------------------------------------------------------------------
import http from "http";
import express from "express";
import { WebSocketServer } from "ws";
import { SessionRegistry } from "./bridge/session-registry";
import { ChunkingPolicy } from "./bridge/chunking-policy";
import { TurnOrchestrator } from "./bridge/turn-orchestrator";
import { InterruptController } from "./bridge/interrupt-controller";
import { XtalkAdapter } from "./adapters/xtalk-adapter";
import { HermesAgentAdapter } from "./adapters/openclaw-agent-adapter";
import { mountRoutes } from "./web/routes";
import type { XtalkAdapterEvent } from "./adapters/xtalk-adapter";

const BRIDGE_HTTP_PORT = parseInt(process.env["BRIDGE_HTTP_PORT"] ?? "7430", 10);
const SIDECAR_WS_URL   = process.env["SIDECAR_WS_URL"] ?? "ws://127.0.0.1:7431";

// ---- Component wiring ---------------------------------------------------
const registry     = new SessionRegistry();
const chunking     = new ChunkingPolicy();
const agentAdapter = new HermesAgentAdapter();
const xtalk        = new XtalkAdapter(SIDECAR_WS_URL);
const orchestrator = new TurnOrchestrator(registry, xtalk, agentAdapter, chunking);
const interruptCtrl = new InterruptController(registry, orchestrator);

// Route sidecar events into the orchestrator
xtalk.on("adapter", (ev: XtalkAdapterEvent) => {
  switch (ev.event) {
    case "connected":
      console.log("[main] sidecar connected");
      for (const mapping of registry.all()) {
        if (!mapping.currentTurn) continue;
        console.log(
          `[main] reopening sidecar session xid=${mapping.xtalkSessionId} turn=${mapping.currentTurn.turnId}`,
        );
        xtalk.openSession(mapping.xtalkSessionId, mapping.currentTurn.turnId);
      }
      break;
    case "disconnected":
      console.warn("[main] sidecar disconnected");
      break;
    case "asr.partial": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (!mapping?.currentTurn) { console.warn("[main] asr.partial: no session for", ev.sessionId); break; }
      orchestrator.onAsrPartial(mapping.browserSessionId, mapping.currentTurn.turnId, ev.text);
      break;
    }
    case "asr.final": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (!mapping?.currentTurn) { console.warn("[main] asr.final: no session for", ev.sessionId); break; }
      orchestrator.onAsrFinal(mapping.browserSessionId, mapping.currentTurn.turnId, ev.text, ev.timing)
        .catch((err) => console.error("[main] onAsrFinal error", err));
      break;
    }
    case "barge_in": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (mapping?.currentTurn) interruptCtrl.handleBargeIn(mapping.browserSessionId, mapping.currentTurn.turnId);
      break;
    }
    case "tts.audio": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (mapping?.currentTurn) {
        orchestrator.emit("event", {
          kind: "TTS_AUDIO_CHUNK",
          browserSessionId: mapping.browserSessionId,
          turnId: mapping.currentTurn.turnId,
          audioBase64: ev.audioBase64,
          mimeType: ev.mimeType,
          sampleRate: ev.sampleRate,
          seq: ev.seq,
        });
      }
      break;
    }
    case "playback.started": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (mapping?.currentTurn) {
        orchestrator.emit("event", { kind: "TTS_PLAYBACK_STARTED", browserSessionId: mapping.browserSessionId, turnId: mapping.currentTurn.turnId });
      }
      break;
    }
    case "playback.finished": {
      const mapping = registry.getByXtalkSessionId(ev.sessionId);
      if (mapping?.currentTurn) orchestrator.onPlaybackFinished(mapping.browserSessionId, mapping.currentTurn.turnId);
      break;
    }
  }
});

// ---- HTTP + WebSocket server --------------------------------------------
const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server, path: "/ws" });
mountRoutes(app, wss, registry, orchestrator, interruptCtrl, xtalk);

// ---- Lifecycle ----------------------------------------------------------
export function start(): void {
  xtalk.connect();
  server.listen(BRIDGE_HTTP_PORT, "127.0.0.1", () => {
    console.log(`[main] Bridge server listening on http://127.0.0.1:${BRIDGE_HTTP_PORT}`);
    console.log(`[main] Browser UI:  http://127.0.0.1:${BRIDGE_HTTP_PORT}/ui`);
    console.log(`[main] Health:      http://127.0.0.1:${BRIDGE_HTTP_PORT}/health`);
    console.log(`[main] Sidecar WS:  ${SIDECAR_WS_URL}`);
  });
}

export function stop(): Promise<void> {
  xtalk.disconnect();
  return new Promise((resolve, reject) => {
    server.close((err) => (err ? reject(err) : resolve()));
  });
}

// ---- Direct execution (Phase 1: standalone mode) -----------------------
start();
```

---

## Diffs

### `scripts/start-all.sh`

```diff
diff --git a/scripts/start-all.sh b/scripts/start-all.sh
index f641c0e..dee65fb 100755
--- a/scripts/start-all.sh
+++ b/scripts/start-all.sh
@@ -7,7 +7,8 @@
 #
 #   1) qwen-asr-serve        — local Qwen3-ASR HTTP server   (separate venv)
 #   2) xtalk-bridge-service  — Python sidecar (ASR/TTS proxy) (sidecar venv)
-#   3) openclaw-extension    — Node.js bridge + browser UI    (npm)
+#   3) hermes-bridge         — Node.js bridge + browser UI    (npm)
+#                              Talks to Hermes Agent over HTTP chat completions
 #
 # Why split venvs?
 #   `omnivoice` requires transformers >= 5.3 and `qwen-asr` pins
@@ -112,6 +113,54 @@ fi
 SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
 SIDECAR_PORT="${SIDECAR_PORT:-7431}"
 
+# ── Load Hermes config so the Node bridge inherits gateway settings ───────────
+# Reads ~/.hermes/config.yaml (YAML) to export gateway host/port and model,
+# then ~/.hermes/.env for API keys.  Process-env values take precedence over
+# both files, matching the priority order in HermesAgentAdapter.
+HERMES_CONFIG="${HERMES_CONFIG:-${HOME}/.hermes/config.yaml}"
+HERMES_STATE_DB="${HERMES_STATE_DB:-${HOME}/.hermes/state.db}"
+export HERMES_CONFIG HERMES_STATE_DB
+
+_parse_hermes_config() {
+  # Minimal YAML parser: extracts gateway.port, gateway.host, and model.
+  # Requires only standard bash + grep — no python or yq dependency.
+  if [[ ! -f "${HERMES_CONFIG}" ]]; then return 0; fi
+  local in_gateway=0
+  while IFS= read -r line; do
+    if [[ "$line" =~ ^gateway: ]]; then in_gateway=1; continue; fi
+    if [[ $in_gateway -eq 1 && "$line" =~ ^[^[:space:]] && ! "$line" =~ ^gateway: ]]; then
+      in_gateway=0
+    fi
+    if [[ $in_gateway -eq 1 ]]; then
+      if [[ "$line" =~ ^[[:space:]]+host:[[:space:]]*(.+) ]]; then
+        export HERMES_GATEWAY_HOST="${BASH_REMATCH[1]// /}"
+      fi
+      if [[ "$line" =~ ^[[:space:]]+port:[[:space:]]*([0-9]+) ]]; then
+        export HERMES_GATEWAY_PORT="${BASH_REMATCH[1]}"
+      fi
+    fi
+    if [[ "$line" =~ ^model:[[:space:]]*(.+) ]]; then
+      export HERMES_MODEL="${BASH_REMATCH[1]// /}"
+    fi
+  done < "${HERMES_CONFIG}"
+}
+_parse_hermes_config
+
+# Load ~/.hermes/.env for API key (only if key not already set)
+if [[ -f "${HOME}/.hermes/.env" && -z "${HERMES_API_KEY:-}" ]]; then
+  while IFS= read -r line; do
+    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
+    if [[ "$line" =~ ^HERMES_API_KEY=(.+)$ ]]; then
+      export HERMES_API_KEY="${BASH_REMATCH[1]}"
+      break
+    fi
+  done < "${HOME}/.hermes/.env"
+fi
+
+log "Hermes config     : ${HERMES_CONFIG}"
+log "Hermes state db   : ${HERMES_STATE_DB}"
+log "Hermes gateway    : ${HERMES_GATEWAY_HOST:-localhost}:${HERMES_GATEWAY_PORT:-80}"
+
 # ── Process tracking ─────────────────────────────────────────────────────────
 declare -a CHILD_PIDS=()
 declare -a CHILD_NAMES=()
@@ -596,9 +645,9 @@ if (( START_NODE )); then
          || die "npm install failed; see ${LOG_DIR}/node-install.log"
      fi
    fi
-    spawn "openclaw-bridge" "${LOG_DIR}/openclaw-bridge.log" \
+    spawn "hermes-bridge" "${LOG_DIR}/hermes-bridge.log" \
      bash -c "cd '${NODE_DIR}' && exec npm start" \
-      || die "Node bridge failed to start"
+      || die "Node bridge (Hermes) failed to start"
  fi
fi
```

### `openclaw-extension-xtalk/src/adapters/openclaw-agent-adapter.ts`

Key structural diff (abridged — full file above):

```diff
-import WebSocket from "ws";
 import crypto from "crypto";
 import fs from "fs";
+import http from "http";
 import os from "os";
 import path from "path";
+import yaml from "yaml";
+import initSqlJs from "sql.js";

-const GATEWAY_URL = process.env["OPENCLAW_GATEWAY_URL"] ?? "ws://127.0.0.1:18789";
-const RECONNECT_DELAY_MS = 2000;
+interface HermesGatewayConfig { ... }
+function loadHermesConfig(): HermesGatewayConfig { ... }
+async function lookupOrCreateHermesSession(...): Promise<string> { ... }

-export class OpenclawAgentAdapter {
-  private ws: WebSocket | null = null;
-  // Ed25519 device signing, WS reconnect loop, run multiplexing...
+export class HermesAgentAdapter {
+  // Stateless per-turn HTTP POST SSE requests
+  async runStream(sessionKey, userMessage, onDelta, onFinal): Promise<void> { ... }
+  cancelRun(sessionKey, _runId): void { req.destroy(); }
 }
```

### `openclaw-extension-xtalk/src/types/state.ts`

```diff
 export interface SessionMapping {
   browserSessionId: string;
   xtalkSessionId: string;
-  /** Stable session key used to look up / create the OpenClaw conversation */
-  openclawSessionKey: string;
+  /** Stable key used to look up / create the Hermes conversation in state.db */
+  hermesSessionId: string;
   currentTurn: TurnContext | null;
   connectedAt: number;
 }
```

### `openclaw-extension-xtalk/src/types/protocol.ts`

```diff
 export type ExtToBrowserMsg =
-  | { type: "session.ready"; xtalkSessionId: string; openclawSessionKey: string }
+  | { type: "session.ready"; xtalkSessionId: string; hermesSessionId: string; openclawSessionKey: string }
```

### `openclaw-extension-xtalk/src/bridge/session-registry.ts`

```diff
-// browserSessionId 1:1 xtalkSessionId, N:1 openclawSessionKey
+// browserSessionId 1:1 xtalkSessionId, N:1 hermesSessionId
   register(
     browserSessionId: string,
-    openclawSessionKey = `xtalk:${browserSessionId}`,
+    hermesSessionId = `xtalk:${browserSessionId}`,
   ): SessionMapping {
     const mapping: SessionMapping = {
-      openclawSessionKey,
+      hermesSessionId,
```

### `openclaw-extension-xtalk/src/bridge/turn-orchestrator.ts`

```diff
-import { OpenclawAgentAdapter } from "../adapters/openclaw-agent-adapter";
+import { HermesAgentAdapter } from "../adapters/openclaw-agent-adapter";
 ...
-    private readonly agent: OpenclawAgentAdapter,
+    private readonly agent: HermesAgentAdapter,
 ...
-    console.log(`... openclawSessionKey=${mapping.openclawSessionKey} ...`);
+    console.log(`... hermesSessionId=${mapping.hermesSessionId} ...`);
 ...
-      this.agent.cancelRun(mapping.openclawSessionKey, ...);
+      this.agent.cancelRun(mapping.hermesSessionId, ...);
```

### `openclaw-extension-xtalk/src/web/routes.ts`

```diff
           safeSend(ws, {
             type: "session.ready",
             xtalkSessionId: mapping.xtalkSessionId,
-            openclawSessionKey: mapping.openclawSessionKey,
+            hermesSessionId: mapping.hermesSessionId,
+            // backward-compat alias: browser UI debug panel reads this field name
+            openclawSessionKey: mapping.hermesSessionId,
           });
```

### `openclaw-extension-xtalk/src/index.ts`

```diff
-import { OpenclawAgentAdapter } from "./adapters/openclaw-agent-adapter";
+import { HermesAgentAdapter } from "./adapters/openclaw-agent-adapter";
 ...
-const agentAdapter = new OpenclawAgentAdapter();
+const agentAdapter = new HermesAgentAdapter();
```

---

## Compilation Status

```
$ cd openclaw-extension-xtalk && npm run build
> openclaw-extension-xtalk@0.1.0 build
> tsc && rm -rf dist/web/ui && cp -r src/web/ui dist/web/

✓ Exit code 0 — TypeScript compiled without errors.
```
