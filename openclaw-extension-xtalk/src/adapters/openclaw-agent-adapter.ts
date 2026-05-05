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
