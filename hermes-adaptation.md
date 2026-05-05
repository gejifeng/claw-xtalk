# Hermes adaptation

## `openclaw-extension-xtalk/src/adapters/openclaw-agent-adapter.ts`

- Replace the `OpenclawAgentAdapter` WebSocket flow (`connect.challenge`, `connect`, `chat.send`, `chat.abort`) with a `HermesAgentAdapter` that calls Hermes over HTTP, e.g. `POST /v1/chat/completions`.
- Remove `.openclaw/identity` loading, device-signing, reconnect handling, and request/run ID bookkeeping that only exists for the OpenClaw gateway protocol.
- Keep the public adapter contract the same: `runStream(sessionId, userMessage, onDelta, onFinal)` should translate Hermes chat-completions output into xtalk deltas/final text; `cancelRun()` should cancel the in-flight HTTP request with a local `AbortController`.
- Rename session fields from `openclawSessionKey` to `hermesSessionId` everywhere the adapter is called.

## `xtalk-bridge-service/xtalk_runtime.py`

- The ASR/TTS pipeline stays the same; the change here is config sourcing, not speech logic.
- Load Hermes defaults from `~/.hermes/config.yaml` before applying env overrides so the runtime can inherit the Hermes API base URL, model, and auth/token settings without any OpenClaw-specific path assumptions.
- Keep xtalk runtime state keyed by `xtalkSessionId`; Hermes conversation/session identity should remain an upstream concern owned by the Node bridge.

## `scripts/start-all.sh`

- Replace OpenClaw-oriented prerequisites with Hermes ones: export `HERMES_CONFIG="${HERMES_CONFIG:-$HOME/.hermes/config.yaml}"` and `HERMES_STATE_DB="${HERMES_STATE_DB:-$HOME/.hermes/state.db}"`.
- Parse `~/.hermes/config.yaml` early so both the sidecar and the Node bridge inherit the same Hermes base URL/model/token settings.
- Update help text/comments to say the Node process talks to Hermes over HTTP chat completions, not to an OpenClaw WebSocket gateway.

## Session mapping

- Current mapping is `browserSessionId -> xtalkSessionId -> openclawSessionKey`, where `openclawSessionKey` is synthesized as `xtalk:${browserSessionId}`.
- Replace that with `browserSessionId -> xtalkSessionId -> hermesSessionId`.
- `hermesSessionId` should be the persistent conversation/session identifier stored by Hermes in `~/.hermes/state.db` (UUID or primary key, depending on Hermes schema).
- On `session.init`, look up or create the Hermes conversation, store its ID in `SessionRegistry`, and reuse that ID on every chat-completions call so multiple xtalk turns stay in one Hermes conversation.
- To survive bridge restarts, keep a stable external key such as `xtalk:${browserSessionId}` in the bridge or in Hermes metadata so the bridge can recover the same `hermesSessionId` from `state.db`.
