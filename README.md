# claw-xtalk

OpenClaw x X-Talk alpha demo for full-duplex browser voice interaction.

This repository wires four pieces into one end-to-end voice loop:

- a browser voice UI for microphone capture and audio playback
- a Node.js bridge that owns turn orchestration and OpenClaw integration
- a Python sidecar that proxies ASR and TTS providers
- an OpenClaw gateway session that remains the single agent authority

The current alpha focuses on a usable demo loop rather than product packaging:

- streaming ASR with partial and final transcripts
- streaming agent output from OpenClaw
- sentence-chunked TTS playback
- full-duplex barge-in during assistant speech
- multi-turn conversation continuity
- browser-side local VAD segmentation to reduce noise-triggered turns
- filler/noise transcript suppression for low-value ASR fragments

## Status

This codebase is at demo alpha quality.

What is already working:

- browser-to-agent-to-speech closed loop
- OpenClaw session reuse across multiple turns
- DashScope Qwen realtime ASR integration
- DashScope CosyVoice TTS integration
- optional local CosyVoice fallback path
- interrupt and playback-stop handling

What is intentionally still rough:

- browser UI is functional, not polished product UI
- deployment is local-process based, not packaged yet
- noise robustness is improved but still demo-grade
- no automated end-to-end test suite yet

## Architecture

```mermaid
flowchart LR
    A[Browser UI] -->|PCM audio / control| B[Node bridge]
    B -->|sidecar protocol| C[Python voice sidecar]
    C -->|realtime ASR| D[Qwen3 ASR]
    C -->|chunked TTS| E[CosyVoice]
    B -->|chat stream| F[OpenClaw gateway]
    F -->|assistant delta/final| B
    C -->|asr.partial asr.final tts.audio| B
    B -->|audio + state| A
```

Design rules:

- OpenClaw remains the only agent and session authority.
- The Python sidecar owns speech-provider integration.
- The Node bridge owns turn state, interruption, and text chunking.
- The browser never talks to cloud ASR/TTS APIs directly.

## Repository Layout

- `docs/`
  architecture notes and official API integration design
- `openclaw-extension-xtalk/`
  Node.js bridge service and browser UI
  - `src/adapters/` provider-facing bridge adapters
  - `src/bridge/` turn orchestration, interruption, and session mapping
  - `src/web/` HTTP routes and in-browser voice UI
  - `package.json` build and runtime entrypoints
- `xtalk-bridge-service/`
  Python speech sidecar
  - `app.py` sidecar entrypoint
  - `websocket_server.py` bridge protocol server
  - `xtalk_runtime.py` ASR and TTS runtime implementations
  - `config/config.py` runtime configuration loading
  - `scripts/bootstrap_cosyvoice.py` optional local CosyVoice bootstrap helper
  - `reference-audio/` local reference material placeholder

## Main Components

### `openclaw-extension-xtalk`

Standalone Node.js bridge process.

Responsibilities:

- host the browser UI at `http://127.0.0.1:7430/ui`
- maintain browser session, speech session, and OpenClaw session mapping
- forward ASR results into OpenClaw
- stream assistant deltas back into TTS chunking
- handle interruption, cancellation, and new-turn rotation

Key modules:

- `src/bridge/turn-orchestrator.ts`
- `src/bridge/session-registry.ts`
- `src/bridge/interrupt-controller.ts`
- `src/adapters/openclaw-agent-adapter.ts`
- `src/adapters/xtalk-adapter.ts`
- `src/web/routes.ts`

### `xtalk-bridge-service`

Python speech sidecar.

Responsibilities:

- accept browser audio relayed by the bridge
- maintain one ASR session per active turn
- proxy realtime ASR events back to the bridge
- serialize TTS requests and stream generated audio chunks back
- translate provider-specific behavior into the project protocol

Supported provider modes:

- `ASR_PROVIDER=qwen-realtime` for DashScope Qwen realtime ASR
- `ASR_PROVIDER=whisper` for local fallback ASR
- `TTS_PROVIDER=aliyun-cosyvoice` for DashScope CosyVoice
- `TTS_PROVIDER=cosyvoice` for local CosyVoice fallback

## Runtime Requirements

Recommended environment:

- Linux
- Node.js 20+
- Python 3.10+
- an OpenClaw gateway available locally
- valid local OpenClaw device identity under `~/.openclaw/identity`
- DashScope API key if using cloud ASR/TTS

Expected local ports:

- `7430` browser UI and bridge HTTP server
- `7431` Python sidecar WebSocket server
- `18789` OpenClaw gateway WebSocket endpoint

## Quick Start

### 1. Install bridge dependencies

```bash
cd openclaw-extension-xtalk
npm install
```

### 2. Create a Python environment for the sidecar

Using conda:

```bash
conda create -n claw-xtalk python=3.10 -y
conda activate claw-xtalk
cd xtalk-bridge-service
pip install -r requirements.txt
```

Using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd xtalk-bridge-service
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file:

```bash
cp xtalk-bridge-service/.env.example xtalk-bridge-service/.env
```

Then fill in at least:

- `DASHSCOPE_API_KEY`
- `ASR_PROVIDER`
- `TTS_PROVIDER`
- `ALIYUN_COSYVOICE_MODEL`
- `ALIYUN_COSYVOICE_VOICE`

For the current demo, the most reliable cloud smoke-test combination is:

- `ASR_PROVIDER=qwen-realtime`
- `QWEN_ASR_MODEL=qwen3-asr-flash-realtime`
- `TTS_PROVIDER=aliyun-cosyvoice`
- `ALIYUN_COSYVOICE_MODEL=cosyvoice-v3-flash`
- `ALIYUN_COSYVOICE_VOICE=longanyang`

Note:

- `cosyvoice-v3.5-flash` does not provide built-in system voices.
- For `cosyvoice-v3.5-flash`, `ALIYUN_COSYVOICE_VOICE` must be a valid clone/design voice ID.

### 4. Make sure OpenClaw is available

The bridge expects an OpenClaw gateway at:

```text
ws://127.0.0.1:18789
```

If your gateway runs elsewhere, set:

```bash
export OPENCLAW_GATEWAY_URL=ws://host:port
```

The bridge also expects authenticated local device identity files under:

```text
~/.openclaw/identity/
```

### 5. Start the Python sidecar

```bash
cd xtalk-bridge-service
python app.py
```

You should see logs similar to:

```text
X-Talk Bridge Service starting up
Configuring Qwen Realtime ASR ...
Configuring DashScope CosyVoice TTS ...
X-Talk sidecar listening on ws://127.0.0.1:7431
```

### 6. Build and start the Node bridge

```bash
cd openclaw-extension-xtalk
npm run build
npm start
```

You should see logs similar to:

```text
Bridge server listening on http://127.0.0.1:7430
Browser UI: http://127.0.0.1:7430/ui
XtalkAdapter connected
OpenclawAgentAdapter connected
```

### 7. Open the browser UI

Open:

```text
http://127.0.0.1:7430/ui
```

Then:

1. allow microphone access
2. start recording
3. speak normally
4. interrupt the assistant while it is talking to test barge-in

## Optional Local CosyVoice Mode

If you want local TTS instead of DashScope:

```bash
cd xtalk-bridge-service
python scripts/bootstrap_cosyvoice.py --install-deps
```

Then switch your `.env` to:

```text
TTS_PROVIDER=cosyvoice
TTS_MODE=zero_shot
```

You may also need to set:

- `COSYVOICE_REPO_DIR`
- `TTS_MODEL_DIR`
- `TTS_PROMPT_WAV`
- `TTS_PROMPT_TEXT`

Notes:

- local CosyVoice is heavier and more environment-sensitive than the cloud path
- cloud mode is the recommended default for demo and GitHub onboarding

## Configuration Reference

### Bridge process

Environment variables consumed by `openclaw-extension-xtalk`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIDGE_HTTP_PORT` | `7430` | HTTP port for browser UI and bridge server |
| `SIDECAR_WS_URL` | `ws://127.0.0.1:7431` | WebSocket address of the Python sidecar |
| `OPENCLAW_GATEWAY_URL` | `ws://127.0.0.1:18789` | OpenClaw gateway endpoint |

### Sidecar process

Configuration is loaded from `xtalk-bridge-service/.env` automatically on startup.

Important variables:

| Variable | Purpose |
| --- | --- |
| `DASHSCOPE_API_KEY` | DashScope credential for Qwen ASR and CosyVoice TTS |
| `ASR_PROVIDER` | `qwen-realtime` or `whisper` |
| `QWEN_ASR_MODEL` | Recommended: `qwen3-asr-flash-realtime` |
| `QWEN_ASR_URL` | Realtime WebSocket endpoint |
| `QWEN_ASR_LANGUAGE` | Recognition language |
| `QWEN_ASR_SAMPLE_RATE` | Usually `16000` |
| `QWEN_ASR_TURN_DETECTION_THRESHOLD` | Server-side VAD sensitivity |
| `QWEN_ASR_TURN_DETECTION_SILENCE_MS` | Server-side endpoint silence window |
| `TTS_PROVIDER` | `aliyun-cosyvoice` or `cosyvoice` |
| `ALIYUN_COSYVOICE_MODEL` | Cloud TTS model |
| `ALIYUN_COSYVOICE_VOICE` | Voice ID or system voice depending on model |
| `ALIYUN_COSYVOICE_AUDIO_FORMAT` | Output format, recommended WAV-compatible PCM |
| `ALIYUN_COSYVOICE_TIMEOUT_MS` | Timeout per TTS request |

## Turn Flow

High-level turn lifecycle:

1. browser captures microphone audio
2. local browser VAD decides when speech actually starts
3. Node bridge opens or refreshes the active turn
4. Python sidecar streams audio to ASR
5. ASR emits `partial` and `final` transcripts
6. bridge filters filler/noise transcripts
7. final user text is sent into the OpenClaw session
8. assistant deltas are chunked into sentence-sized TTS work items
9. sidecar synthesizes audio chunk by chunk
10. browser plays audio and supports user barge-in
11. on playback completion, the bridge rotates into a new turn automatically

## Current Demo Behaviors

Implemented behaviors worth knowing before debugging:

- assistant final text is not replayed twice
- normal playback completion creates a fresh new turn
- browser mic upload is segmented locally instead of raw continuous upload
- pure filler transcripts such as simple `嗯` fragments are suppressed before they hit the agent
- browser and sidecar both contribute to interruption detection

## Known Limitations

- very short but valid-looking fragments can still slip through ASR as low-information turns
- local VAD thresholds may need retuning across microphones and rooms
- this project currently assumes a local OpenClaw gateway with working device identity
- packaging, installer scripts, screenshots, and CI are not finalized yet

## Documents

Design documents in `docs/`:

- `docs/openclaw-xtalk-phase1-architecture.md`
- `docs/qwen3-asr-cosyvoice3-official-api-design.md`

Recommended reading order:

1. architecture doc
2. official API migration doc
3. this README for actual repository usage

## Roadmap After Alpha

- package the bridge as a proper OpenClaw extension artifact
- split provider adapters into cleaner modules
- add transcript quality metrics and better noise rejection
- add repeatable smoke tests for ASR and TTS
- support remote deployment of the speech sidecar

## License

Apache-2.0. See `LICENSE`.
