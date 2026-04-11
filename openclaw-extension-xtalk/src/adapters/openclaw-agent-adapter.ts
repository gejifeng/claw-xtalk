// -----------------------------------------------------------------------
// OpenclawAgentAdapter – connects to the OpenClaw gateway over WebSocket,
// using the local device identity for authenticated access.
// -----------------------------------------------------------------------
import WebSocket from "ws";
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";

const GATEWAY_URL = process.env["OPENCLAW_GATEWAY_URL"] ?? "ws://127.0.0.1:18789";
const RECONNECT_DELAY_MS = 2000;

interface PendingRun {
  sessionKey: string;
  onDelta: (delta: string) => void;
  onFinal: (finalText: string) => void;
  resolve: () => void;
  reject: (err: Error) => void;
  prevLen: number;
}

export class OpenclawAgentAdapter {
  private ws: WebSocket | null = null;
  private connected = false;
  private connectWaiters: Array<() => void> = [];
  private pendingRuns = new Map<string, PendingRun>();
  // reqId -> provisional run before chat.send returns the authoritative runId.
  private pendingReqIds = new Map<string, PendingRun>();
  private activeRunIdsBySession = new Map<string, string>();
  private reqSeq = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly deviceId: string;
  private readonly privateKeyPem: string;
  private readonly publicKeyB64url: string;
  private readonly deviceToken: string;
  private readonly scopes: string[];

  constructor() {
    const identityDir = path.join(os.homedir(), ".openclaw", "identity");
    const deviceJson = JSON.parse(
      fs.readFileSync(path.join(identityDir, "device.json"), "utf8"),
    ) as { deviceId: string; privateKeyPem: string; publicKeyPem: string };
    const authJson = JSON.parse(
      fs.readFileSync(path.join(identityDir, "device-auth.json"), "utf8"),
    ) as { tokens: { operator: { token: string; scopes: string[] } } };

    this.deviceId = deviceJson.deviceId;
    this.privateKeyPem = deviceJson.privateKeyPem;
    this.publicKeyB64url = extractEd25519RawB64url(deviceJson.publicKeyPem);
    this.deviceToken = authJson.tokens.operator.token;
    this.scopes = authJson.tokens.operator.scopes;

    this.connect();
  }

  private connect(): void {
    const ws = new WebSocket(GATEWAY_URL);
    this.ws = ws;

    ws.on("open", () => {
      console.log("[OpenclawAgentAdapter] WS open");
    });

    ws.on("message", (data: Buffer) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(data.toString()) as Record<string, unknown>;
      } catch {
        return;
      }
      this.handleMessage(msg, ws);
    });

    ws.on("close", () => {
      console.warn("[OpenclawAgentAdapter] WS closed");
      this.connected = false;
      this.ws = null;
      for (const [, run] of this.pendingRuns) {
        run.reject(new Error("gateway connection closed"));
      }
      this.pendingRuns.clear();
      for (const [, run] of this.pendingReqIds) {
        run.reject(new Error("gateway connection closed"));
      }
      this.pendingReqIds.clear();
      this.activeRunIdsBySession.clear();
      this.scheduleReconnect();
    });

    ws.on("error", (err: Error) => {
      console.error("[OpenclawAgentAdapter] WS error:", err.message);
    });
  }

  private handleMessage(msg: Record<string, unknown>, ws: WebSocket): void {
    const type = msg["type"];

    // Challenge → sign and send connect request
    if (type === "event" && msg["event"] === "connect.challenge") {
      const payload = msg["payload"] as { nonce: string };
      const nonce = payload.nonce;
      const signedAtMs = Date.now();
      const sigPayload = [
        "v3",
        this.deviceId,
        "cli",
        "cli",
        "operator",
        this.scopes.join(","),
        String(signedAtMs),
        this.deviceToken,
        nonce,
        "linux",
        "",
      ].join("|");
      const key = crypto.createPrivateKey(this.privateKeyPem);
      const sigBuf = crypto.sign(null, Buffer.from(sigPayload, "utf8"), key);
      const signature = sigBuf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
      ws.send(
        JSON.stringify({
          type: "req",
          id: "oc-connect",
          method: "connect",
          params: {
            minProtocol: 3,
            maxProtocol: 3,
            client: { id: "cli", version: "0.1.0", platform: "linux", mode: "cli" },
            auth: { deviceToken: this.deviceToken },
            device: {
              id: this.deviceId,
              publicKey: this.publicKeyB64url,
              signature,
              signedAt: signedAtMs,
              nonce,
            },
            scopes: this.scopes,
            role: "operator",
          },
        }),
      );
    }

    // Connect response
    if (type === "res" && msg["id"] === "oc-connect") {
      if (msg["ok"]) {
        const pl = msg["payload"] as { auth?: { scopes?: string[] } } | undefined;
        console.log("[OpenclawAgentAdapter] connected scopes:", pl?.auth?.scopes);
        this.connected = true;
        const waiters = this.connectWaiters.splice(0);
        for (const w of waiters) w();
      } else {
        console.error("[OpenclawAgentAdapter] connect rejected:", JSON.stringify(msg["error"]));
        ws.close();
      }
    }

    // chat.send ACK – detect failures early
    if (type === "res" && msg["id"] !== "oc-connect") {
      const reqId = msg["id"] as string;
      const pendingRun = this.pendingReqIds.get(reqId);
      if (pendingRun) {
        this.pendingReqIds.delete(reqId);
        if (!msg["ok"]) {
          console.error(
            `[OpenclawAgentAdapter] chat.send rejected sessionKey=${pendingRun.sessionKey}:`,
            JSON.stringify(msg["error"]),
          );
          pendingRun.reject(new Error(`chat.send rejected: ${JSON.stringify(msg["error"])}`));
        } else {
          const payload = msg["payload"] as { runId?: string; status?: string } | undefined;
          const runId = payload?.runId;
          if (!runId) {
            pendingRun.reject(new Error("chat.send ack missing runId"));
            return;
          }
          this.pendingRuns.set(runId, pendingRun);
          this.activeRunIdsBySession.set(pendingRun.sessionKey, runId);
          console.log(
            `[OpenclawAgentAdapter] chat.send ack ok sessionKey=${pendingRun.sessionKey} runId=${runId} status=${payload?.status}`,
          );
        }
      }
    }

    // Streaming chat events
    if (type === "event" && msg["event"] === "chat") {
      const p = msg["payload"] as {
        runId: string;
        sessionKey: string;
        state: string;
        message?: { content?: Array<{ text?: string }> };
      };
      const run = this.pendingRuns.get(p.runId);
      if (!run) return;

      const text = p.message?.content?.[0]?.text ?? "";

      if (p.state === "delta") {
        const newChars = text.slice(run.prevLen);
        if (newChars) {
          run.prevLen = text.length;
          run.onDelta(newChars);
        }
      } else if (p.state === "final") {
        console.log(
          `[OpenclawAgentAdapter] chat final sessionKey=${p.sessionKey} runId=${p.runId} len=${text.length}`,
        );
        this.pendingRuns.delete(p.runId);
        this.activeRunIdsBySession.delete(run.sessionKey);
        run.onFinal(text);
        run.resolve();
      } else if (p.state === "aborted" || p.state === "error") {
        console.warn(
          `[OpenclawAgentAdapter] chat ${p.state} sessionKey=${p.sessionKey} runId=${p.runId}`,
        );
        this.pendingRuns.delete(p.runId);
        this.activeRunIdsBySession.delete(run.sessionKey);
        run.onFinal(text);
        run.resolve();
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, RECONNECT_DELAY_MS);
  }

  private waitForConnection(): Promise<void> {
    if (this.connected) return Promise.resolve();
    return new Promise<void>((resolve) => this.connectWaiters.push(resolve));
  }

  private nextId(): string {
    return `oc-req-${++this.reqSeq}`;
  }

  async runStream(
    sessionKey: string,
    userMessage: string,
    onDelta: (delta: string) => void,
    onFinal: (finalText: string) => void,
  ): Promise<void> {
    await this.waitForConnection();

    const idempotencyKey = crypto.randomUUID();
    const reqId = this.nextId();

    console.log(`[OpenclawAgentAdapter] chat.send sessionKey=${sessionKey} reqId=${reqId} msg="${userMessage.slice(0, 60)}"`);

    return new Promise<void>((resolve, reject) => {
      const pendingRun: PendingRun = {
        sessionKey,
        onDelta,
        onFinal,
        resolve,
        reject,
        prevLen: 0,
      };
      this.pendingReqIds.set(reqId, pendingRun);
      this.ws!.send(
        JSON.stringify({
          type: "req",
          id: reqId,
          method: "chat.send",
          params: { sessionKey, message: userMessage, idempotencyKey },
        }),
      );
    });
  }

  cancelRun(sessionKey: string, _runId: string): void {
    if (!this.connected || !this.ws) return;
    const runId = this.activeRunIdsBySession.get(sessionKey);
    this.ws.send(
      JSON.stringify({
        type: "req",
        id: this.nextId(),
        method: "chat.abort",
        params: runId ? { sessionKey, runId } : { sessionKey },
      }),
    );
  }
}

/** Extract the raw 32-byte Ed25519 public key from a SubjectPublicKeyInfo PEM and encode as base64url. */
function extractEd25519RawB64url(pem: string): string {
  const b64 = pem.replace(/-----[^-]+-----/g, "").replace(/\s/g, "");
  const der = Buffer.from(b64, "base64");
  // SPKI for Ed25519: 12-byte header + 32-byte raw key
  const raw = der.slice(der.length - 32);
  return raw.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
}
