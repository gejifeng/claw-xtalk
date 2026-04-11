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
import { OpenclawAgentAdapter } from "./adapters/openclaw-agent-adapter";
import { mountRoutes } from "./web/routes";
import type { XtalkAdapterEvent } from "./adapters/xtalk-adapter";

const BRIDGE_HTTP_PORT = parseInt(process.env["BRIDGE_HTTP_PORT"] ?? "7430", 10);
const SIDECAR_WS_URL   = process.env["SIDECAR_WS_URL"] ?? "ws://127.0.0.1:7431";

// ---- Component wiring ---------------------------------------------------
const registry     = new SessionRegistry();
const chunking     = new ChunkingPolicy();
const agentAdapter = new OpenclawAgentAdapter();
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
