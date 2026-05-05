#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Claw-Xtalk one-click launcher (split-process mode, fully auto).
#
# Brings up three long-running components in dependency order and tears them
# down cleanly on Ctrl-C / SIGTERM:
#
#   1) qwen-asr-serve        — local Qwen3-ASR HTTP server   (separate venv)
#   2) xtalk-bridge-service  — Python sidecar (ASR/TTS proxy) (sidecar venv)
#   3) hermes-bridge         — Node.js bridge + browser UI    (npm)
#                              Talks to Hermes Agent over HTTP chat completions
#
# Why split venvs?
#   `omnivoice` requires transformers >= 5.3 and `qwen-asr` pins
#   transformers == 4.57.6, so the two cannot coexist in one Python env.
#   The sidecar talks to qwen-asr-serve over its OpenAI-compatible HTTP API
#   to keep both engines reachable at once.
#
# Everything is auto-detected. There are no setup phases — just run:
#
#   ./scripts/start-all.sh                 # bring up everything
#   ./scripts/start-all.sh --no-asr        # skip the local ASR server (use cloud)
#   ./scripts/start-all.sh --no-node       # skip the Node bridge
#   ./scripts/start-all.sh --reinstall     # force reinstall both venvs
#   ./scripts/start-all.sh --help
# ─────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
SIDECAR_DIR="${ROOT_DIR}/xtalk-bridge-service"
NODE_DIR="${ROOT_DIR}/openclaw-extension-xtalk"
LOG_DIR="${ROOT_DIR}/logs"
RUN_DIR="${ROOT_DIR}/.run"
mkdir -p "${LOG_DIR}" "${RUN_DIR}"

SIDECAR_VENV="${SIDECAR_VENV:-${ROOT_DIR}/.venv}"
ASR_VENV="${ASR_VENV:-${ROOT_DIR}/.venv-qwen-asr}"

# ── Defaults (overridable via env or .env) ───────────────────────────────────
ASR_MODEL="${QWEN_LOCAL_MODEL:-Qwen/Qwen3-ASR-0.6B}"
ASR_HOST="${QWEN_ASR_SERVE_HOST:-127.0.0.1}"
ASR_PORT="${QWEN_ASR_SERVE_PORT:-8910}"
# ── Low-VRAM defaults so a TTS engine (omnivoice / cosyvoice) can coexist on
# the same GPU. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/
#
# `gpu-memory-utilization` is the *cap* on vLLM's GPU footprint, not a target.
# 0.40 keeps Qwen3-ASR-0.6B inside ~40% of the card and leaves the rest for
# TTS + framework overhead. Raise it if you only run ASR.
ASR_GPU_MEM_UTIL="${QWEN_ASR_SERVE_GPU_MEM_UTIL:-0.40}"
# The model advertises max_model_len=65536 which would reserve ~7 GiB of KV
# cache. ASR turns are short audio (< 30 s), so 2048 tokens is plenty.
ASR_MAX_MODEL_LEN="${QWEN_ASR_SERVE_MAX_MODEL_LEN:-2048}"
# Single-stream ASR — no need to reserve KV slots for 256 concurrent requests.
ASR_MAX_NUM_SEQS="${QWEN_ASR_SERVE_MAX_NUM_SEQS:-1}"
# Skip CUDA-graph capture (saves 1–3 GiB at the cost of ~10–20% throughput,
# which is irrelevant for a single-user real-time ASR session).
ASR_ENFORCE_EAGER="${QWEN_ASR_SERVE_ENFORCE_EAGER:-1}"
# fp8 KV cache halves KV memory with negligible WER impact for ASR. Set to
# "auto" to disable (e.g. on Pascal/Volta GPUs without FP8 support).
ASR_KV_CACHE_DTYPE="${QWEN_ASR_SERVE_KV_CACHE_DTYPE:-fp8}"
# No CPU swap reservation — we never want vLLM paging KV to host RAM here.
ASR_SWAP_SPACE="${QWEN_ASR_SERVE_SWAP_SPACE:-0}"
ASR_EXTRA_ARGS="${QWEN_ASR_SERVE_EXTRA_ARGS:-}"

START_ASR=1
START_SIDECAR=1
START_NODE=1
FORCE_REINSTALL=0

# ── Pretty logging ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
  C_RESET=""; C_DIM=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
fi
log()  { printf '%s[start-all]%s %s\n' "${C_CYAN}" "${C_RESET}" "$*"; }
ok()   { printf '%s[start-all]%s %s%s%s\n' "${C_CYAN}" "${C_RESET}" "${C_GREEN}" "$*" "${C_RESET}"; }
warn() { printf '%s[start-all]%s %s%s%s\n' "${C_CYAN}" "${C_RESET}" "${C_YELLOW}" "$*" "${C_RESET}" >&2; }
err()  { printf '%s[start-all]%s %s%s%s\n' "${C_CYAN}" "${C_RESET}" "${C_RED}"   "$*" "${C_RESET}" >&2; }
die()  { err "$*"; exit 1; }
hint() { printf '%s[start-all]%s   %s%s%s\n' "${C_CYAN}" "${C_RESET}" "${C_DIM}" "$*" "${C_RESET}" >&2; }

usage() {
  sed -n '2,/^# ─────/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

# ── Parse args ───────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --no-asr)     START_ASR=0 ;;
    --no-sidecar) START_SIDECAR=0 ;;
    --no-node)    START_NODE=0 ;;
    --reinstall)  FORCE_REINSTALL=1 ;;
    -h|--help)    usage ;;
    *) die "unknown arg: $arg (run with --help)" ;;
  esac
done

# ── Load sidecar .env so we know which ports to wait on ──────────────────────
if [[ -f "${SIDECAR_DIR}/.env" ]]; then
  set -a
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    eval "export $line" 2>/dev/null || true
  done < "${SIDECAR_DIR}/.env"
  set +a
fi
SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
SIDECAR_PORT="${SIDECAR_PORT:-7431}"

# ── Load Hermes config so the Node bridge inherits gateway settings ───────────
# Reads ~/.hermes/config.yaml (YAML) to export gateway host/port and model,
# then ~/.hermes/.env for API keys.  Process-env values take precedence over
# both files, matching the priority order in HermesAgentAdapter.
HERMES_CONFIG="${HERMES_CONFIG:-${HOME}/.hermes/config.yaml}"
HERMES_STATE_DB="${HERMES_STATE_DB:-${HOME}/.hermes/state.db}"
export HERMES_CONFIG HERMES_STATE_DB

_parse_hermes_config() {
  # Minimal YAML parser: extracts gateway.port, gateway.host, and model.
  # Requires only standard bash + grep — no python or yq dependency.
  if [[ ! -f "${HERMES_CONFIG}" ]]; then return 0; fi
  local in_gateway=0
  while IFS= read -r line; do
    if [[ "$line" =~ ^gateway: ]]; then in_gateway=1; continue; fi
    if [[ $in_gateway -eq 1 && "$line" =~ ^[^[:space:]] && ! "$line" =~ ^gateway: ]]; then
      in_gateway=0
    fi
    if [[ $in_gateway -eq 1 ]]; then
      if [[ "$line" =~ ^[[:space:]]+host:[[:space:]]*(.+) ]]; then
        export HERMES_GATEWAY_HOST="${BASH_REMATCH[1]// /}"
      fi
      if [[ "$line" =~ ^[[:space:]]+port:[[:space:]]*([0-9]+) ]]; then
        export HERMES_GATEWAY_PORT="${BASH_REMATCH[1]}"
      fi
    fi
    if [[ "$line" =~ ^model:[[:space:]]*(.+) ]]; then
      export HERMES_MODEL="${BASH_REMATCH[1]// /}"
    fi
  done < "${HERMES_CONFIG}"
}
_parse_hermes_config

# Load ~/.hermes/.env for API key (only if key not already set)
if [[ -f "${HOME}/.hermes/.env" && -z "${HERMES_API_KEY:-}" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^HERMES_API_KEY=(.+)$ ]]; then
      export HERMES_API_KEY="${BASH_REMATCH[1]}"
      break
    fi
  done < "${HOME}/.hermes/.env"
fi

log "Hermes config     : ${HERMES_CONFIG}"
log "Hermes state db   : ${HERMES_STATE_DB}"
log "Hermes gateway    : ${HERMES_GATEWAY_HOST:-localhost}:${HERMES_GATEWAY_PORT:-80}"

# ── Process tracking ─────────────────────────────────────────────────────────
declare -a CHILD_PIDS=()
declare -a CHILD_NAMES=()
SHUTTING_DOWN=0

cleanup() {
  local exit_code=$?
  if (( SHUTTING_DOWN )); then return; fi
  SHUTTING_DOWN=1
  echo
  log "shutting down (exit_code=${exit_code}) ..."
  for ((i=${#CHILD_PIDS[@]}-1; i>=0; i--)); do
    local pid="${CHILD_PIDS[$i]}"
    local name="${CHILD_NAMES[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      log "  stopping ${name} (pid=${pid})"
      kill -TERM "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  local deadline=$(( $(date +%s) + 10 ))
  for pid in "${CHILD_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < deadline )); do
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      warn "  force-killing pid=${pid}"
      kill -KILL "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -f "${RUN_DIR}"/*.pid 2>/dev/null || true
  log "all components stopped"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

on_error() {
  local exit_code=$?
  local line=$1
  err "failed at line ${line} (exit=${exit_code})"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

# ── System helpers ───────────────────────────────────────────────────────────
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1 ${2:-}"; }
have_cmd()    { command -v "$1" >/dev/null 2>&1; }

PIP_INSTALLER=""
PYTHON_BIN=""

detect_python() {
  for cand in python3 python; do
    if have_cmd "$cand"; then PYTHON_BIN="$(command -v "$cand")"; return; fi
  done
  die "no python3 on PATH"
}

python_has_venv() {
  "$PYTHON_BIN" -c "import venv, ensurepip" >/dev/null 2>&1
}

apt_hint_for_venv() {
  if have_cmd apt-get; then
    local ver
    ver="$("$PYTHON_BIN" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    if [[ -n "$ver" ]]; then
      hint "On Debian/Ubuntu run: sudo apt-get install -y python${ver}-venv python${ver}-dev"
    else
      hint "On Debian/Ubuntu run: sudo apt-get install -y python3-venv python3-dev"
    fi
  fi
  hint "Or install uv (https://docs.astral.sh/uv/) and re-run this script."
}

UV_BIN=""

# Make sure $HOME/.local/bin and the uv installer's cargo path are on PATH so
# we can pick up a freshly installed `uv` without requiring a shell reload.
ensure_uv_on_path() {
  local extra
  for extra in "${HOME}/.local/bin" "${HOME}/.cargo/bin"; do
    if [[ -d "$extra" ]] && [[ ":$PATH:" != *":${extra}:"* ]]; then
      export PATH="${extra}:${PATH}"
    fi
  done
}

install_uv() {
  log "installing 'uv' (https://docs.astral.sh/uv/) ..."
  local installer_log="${LOG_DIR}/uv-install.log"
  if have_cmd curl; then
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh >>"${installer_log}" 2>&1; then
      err "uv install via curl failed; see ${installer_log}"
      tail -n 40 "${installer_log}" >&2 || true
      return 1
    fi
  elif have_cmd wget; then
    if ! wget -qO- https://astral.sh/uv/install.sh | sh >>"${installer_log}" 2>&1; then
      err "uv install via wget failed; see ${installer_log}"
      tail -n 40 "${installer_log}" >&2 || true
      return 1
    fi
  else
    err "neither curl nor wget available; cannot auto-install uv"
    return 1
  fi
  ensure_uv_on_path
  if ! have_cmd uv; then
    err "uv installed but not on PATH; expected ${HOME}/.local/bin/uv"
    return 1
  fi
  ok "uv installed: $(command -v uv)  ($(uv --version 2>/dev/null || echo unknown))"
  return 0
}

choose_installer() {
  ensure_uv_on_path
  if ! have_cmd uv; then
    if ! install_uv; then
      die "could not install 'uv'. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
  fi
  UV_BIN="$(command -v uv)"
  PIP_INSTALLER="uv"
  log "using uv at ${UV_BIN} ($(uv --version 2>/dev/null || echo unknown))"
}

venv_is_complete() {
  local venv=$1
  [[ -f "${venv}/bin/activate" ]] && [[ -x "${venv}/bin/python" || -x "${venv}/bin/python3" ]]
}

create_venv() {
  local venv=$1 logfile=$2
  if [[ -d "$venv" ]] && ! venv_is_complete "$venv"; then
    warn "found broken venv at ${venv} (no bin/activate); removing"
    rm -rf "$venv"
  fi
  if venv_is_complete "$venv" 2>/dev/null; then return 0; fi

  log "creating venv at ${venv}"
  if [[ "$PIP_INSTALLER" == "uv" ]]; then
    if ! uv venv "$venv" --python "$PYTHON_BIN" >>"$logfile" 2>&1; then
      err "uv venv failed; see ${logfile}"
      tail -n 40 "$logfile" >&2 || true
      return 1
    fi
  else
    if ! python_has_venv; then
      err "Python's 'venv' / 'ensurepip' module is missing for ${PYTHON_BIN}."
      apt_hint_for_venv
      return 1
    fi
    if ! "$PYTHON_BIN" -m venv "$venv" >>"$logfile" 2>&1; then
      err "python -m venv failed; see ${logfile}"
      tail -n 40 "$logfile" >&2 || true
      apt_hint_for_venv
      return 1
    fi
  fi

  if ! venv_is_complete "$venv"; then
    err "venv at ${venv} is incomplete after creation"
    apt_hint_for_venv
    rm -rf "$venv"
    return 1
  fi
  ok "venv ready at ${venv}"
  return 0
}

venv_pip_install() {
  local venv=$1 logfile=$2; shift 2
  local py="${venv}/bin/python"
  if [[ "$PIP_INSTALLER" == "uv" ]]; then
    log "  uv pip install $*"
    if ! VIRTUAL_ENV="$venv" uv pip install --python "$py" "$@" >>"$logfile" 2>&1; then
      err "uv pip install failed; see ${logfile}"
      tail -n 40 "$logfile" >&2 || true
      return 1
    fi
  else
    log "  pip install $*"
    if ! "$py" -m pip install --upgrade pip wheel >>"$logfile" 2>&1; then
      warn "  pip self-upgrade failed (continuing)"
    fi
    if ! "$py" -m pip install "$@" >>"$logfile" 2>&1; then
      err "pip install failed; see ${logfile}"
      tail -n 40 "$logfile" >&2 || true
      return 1
    fi
  fi
  return 0
}

venv_has_module() {
  local venv=$1 mod=$2
  "${venv}/bin/python" -c "import ${mod}" >/dev/null 2>&1
}

# ── Network helpers ──────────────────────────────────────────────────────────
port_is_free() {
  local host=$1 port=$2
  if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

wait_for_port() {
  local host=$1 port=$2 name=$3 timeout=${4:-180}
  log "waiting for ${name} on ${host}:${port} (timeout ${timeout}s)"
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      ok "${name} is up on ${host}:${port}"
      return 0
    fi
    for ((i=0; i<${#CHILD_PIDS[@]}; i++)); do
      if [[ "${CHILD_NAMES[$i]}" == "${name}" ]]; then
        if ! kill -0 "${CHILD_PIDS[$i]}" 2>/dev/null; then
          err "${name} exited before opening ${host}:${port}"
          err "tail of log:"; tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
          return 1
        fi
      fi
    done
    sleep 0.5
  done
  err "${name} did not open ${host}:${port} within ${timeout}s"
  err "tail of log:"; tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
  return 1
}

wait_for_http() {
  local url=$1 name=$2 timeout=${3:-180}
  log "waiting for ${name} HTTP ${url} (timeout ${timeout}s)"
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if curl -fsS -m 2 -o /dev/null "${url}"; then
      ok "${name} HTTP ready"
      return 0
    fi
    for ((i=0; i<${#CHILD_PIDS[@]}; i++)); do
      if [[ "${CHILD_NAMES[$i]}" == "${name}" ]]; then
        if ! kill -0 "${CHILD_PIDS[$i]}" 2>/dev/null; then
          err "${name} exited before responding on ${url}"
          err "tail of log:"; tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
          return 1
        fi
      fi
    done
    sleep 1
  done
  err "${name} HTTP not ready within ${timeout}s — ${url}"
  err "tail of log:"; tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
  return 1
}

http_is_healthy() {
  local url=$1
  curl -fsS -m 2 -o /dev/null "$url" >/dev/null 2>&1
}

# ── Process helper ───────────────────────────────────────────────────────────
spawn() {
  local name=$1 logfile=$2; shift 2
  log "starting ${name}"
  : > "${logfile}"
  setsid bash -c "exec \"\$@\" >>'${logfile}' 2>&1" _ "$@" &
  local pid=$!
  CHILD_PIDS+=("$pid")
  CHILD_NAMES+=("$name")
  echo "$pid" > "${RUN_DIR}/${name}.pid"
  log "  ${name} pid=${pid}  log=${logfile}"
  sleep 1.5
  if ! kill -0 "$pid" 2>/dev/null; then
    err "${name} exited immediately. tail of log:"
    tail -n 80 "${logfile}" >&2 || true
    return 1
  fi
  return 0
}

# ── GPU / CUDA detection (non-fatal) ─────────────────────────────────────────
HAS_GPU=0
detect_gpu() {
  if have_cmd nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  else
    HAS_GPU=0
    warn "no NVIDIA GPU detected; will install bare qwen-asr (transformers backend, slower)"
  fi
}

# ── Sidecar venv setup ───────────────────────────────────────────────────────
ensure_sidecar_venv() {
  local logfile="${LOG_DIR}/sidecar-install.log"
  if (( FORCE_REINSTALL )); then rm -rf "$SIDECAR_VENV"; fi
  create_venv "$SIDECAR_VENV" "$logfile" || die "could not prepare sidecar venv"

  local need_install=0
  for mod in websockets numpy faster_whisper huggingface_hub dashscope; do
    if ! venv_has_module "$SIDECAR_VENV" "$mod"; then need_install=1; break; fi
  done
  if (( need_install )); then
    log "installing sidecar requirements (this may take a while)"
    venv_pip_install "$SIDECAR_VENV" "$logfile" -r "${SIDECAR_DIR}/requirements.txt" \
      || die "sidecar pip install failed"
    ok "sidecar requirements installed"
  else
    log "sidecar venv already has core deps; skipping pip install"
  fi
}

# ── ASR venv setup ───────────────────────────────────────────────────────────
ensure_asr_venv() {
  local logfile="${LOG_DIR}/qwen-asr-install.log"
  if (( FORCE_REINSTALL )); then rm -rf "$ASR_VENV"; fi
  create_venv "$ASR_VENV" "$logfile" || die "could not prepare qwen-asr venv"

  if [[ -x "${ASR_VENV}/bin/qwen-asr-serve" ]]; then
    log "qwen-asr-serve already installed in ${ASR_VENV}; skipping"
    return 0
  fi

  log "installing qwen-asr into ${ASR_VENV} (this can take several minutes)"
  if (( HAS_GPU )); then
    if ! venv_pip_install "$ASR_VENV" "$logfile" -U "qwen-asr[vllm]"; then
      warn "qwen-asr[vllm] install failed; falling back to bare qwen-asr (no streaming)"
      venv_pip_install "$ASR_VENV" "$logfile" -U "qwen-asr" \
        || die "qwen-asr install failed; see ${logfile}"
    fi
  else
    venv_pip_install "$ASR_VENV" "$logfile" -U "qwen-asr" \
      || die "qwen-asr install failed; see ${logfile}"
  fi

  if [[ ! -x "${ASR_VENV}/bin/qwen-asr-serve" ]]; then
    err "qwen-asr-serve not found in ${ASR_VENV}/bin after install"
    err "tail of install log:"; tail -n 80 "$logfile" >&2 || true
    die "qwen-asr installation incomplete"
  fi
  ok "qwen-asr-serve installed"
}

# ── Model bootstrap ──────────────────────────────────────────────────────────
ensure_asr_model() {
  local target="${SIDECAR_DIR}/pretrained_models/$(basename "${ASR_MODEL}")"
  if [[ -d "${target}" && -n "$(ls -A "${target}" 2>/dev/null || true)" ]]; then
    log "ASR model already present at ${target}"
    return 0
  fi
  log "fetching ASR model ${ASR_MODEL} -> ${target}"
  if ! venv_has_module "$SIDECAR_VENV" huggingface_hub; then
    venv_pip_install "$SIDECAR_VENV" "${LOG_DIR}/sidecar-install.log" -U huggingface-hub \
      || warn "huggingface-hub install failed; qwen-asr will try to download at runtime"
  fi
  if ! "${SIDECAR_VENV}/bin/python" "${SIDECAR_DIR}/scripts/bootstrap_qwen3_asr.py" \
        --model "${ASR_MODEL}" \
        --target "${target}" \
        2>>"${LOG_DIR}/asr-model-fetch.log"; then
    warn "model bootstrap failed; qwen-asr will try to download at runtime"
    warn "see ${LOG_DIR}/asr-model-fetch.log for details"
  fi
}

# ── Phase 0: detect ──────────────────────────────────────────────────────────
require_cmd curl
detect_python
choose_installer
log "root dir       : ${ROOT_DIR}"
log "python         : ${PYTHON_BIN} ($($PYTHON_BIN -c 'import platform;print(platform.python_version())' 2>/dev/null || echo unknown))"
log "sidecar venv   : ${SIDECAR_VENV}"
log "asr venv       : ${ASR_VENV}"
log "logs dir       : ${LOG_DIR}"

if (( START_ASR )); then detect_gpu; fi

# ── Phase 1: install (idempotent) ────────────────────────────────────────────
if (( START_SIDECAR )); then ensure_sidecar_venv; fi

REUSE_ASR=0
if (( START_ASR )); then
  if ! port_is_free "${ASR_HOST}" "${ASR_PORT}" \
     && http_is_healthy "http://${ASR_HOST}:${ASR_PORT}/v1/models"; then
    REUSE_ASR=1
    warn "qwen-asr-serve already healthy on ${ASR_HOST}:${ASR_PORT}; reusing existing instance"
  else
    ensure_asr_venv
    ensure_asr_model
  fi
fi

# ── Phase 2: launch qwen-asr-serve ───────────────────────────────────────────
ASR_URL_BASE="http://${ASR_HOST}:${ASR_PORT}/v1"
if (( START_ASR )) && (( ! REUSE_ASR )); then
  if ! port_is_free "${ASR_HOST}" "${ASR_PORT}"; then
    die "port ${ASR_HOST}:${ASR_PORT} is in use but not responding healthily; free it or set QWEN_ASR_SERVE_PORT"
  fi
  ASR_BIN="${ASR_VENV}/bin/qwen-asr-serve"
  [[ -x "${ASR_BIN}" ]] || die "missing ${ASR_BIN}; rerun with --reinstall"

  # Build the conserving-memory flag list. Each flag is only injected when the
  # user did NOT already pass an equivalent one in QWEN_ASR_SERVE_EXTRA_ARGS,
  # so power-users can override individual knobs without losing the rest.
  ASR_MEM_ARGS=()
  _has_arg() {
    local needle=$1
    [[ "${ASR_EXTRA_ARGS}" == *"${needle}"* ]]
  }
  if ! _has_arg "--max-model-len" && ! _has_arg "--max_model_len"; then
    ASR_MEM_ARGS+=(--max-model-len "${ASR_MAX_MODEL_LEN}")
  fi
  if ! _has_arg "--max-num-seqs" && ! _has_arg "--max_num_seqs"; then
    ASR_MEM_ARGS+=(--max-num-seqs "${ASR_MAX_NUM_SEQS}")
  fi
  if (( ASR_ENFORCE_EAGER )) && ! _has_arg "--enforce-eager"; then
    ASR_MEM_ARGS+=(--enforce-eager)
  fi
  if [[ "${ASR_KV_CACHE_DTYPE}" != "auto" ]] \
     && ! _has_arg "--kv-cache-dtype" && ! _has_arg "--kv_cache_dtype"; then
    ASR_MEM_ARGS+=(--kv-cache-dtype "${ASR_KV_CACHE_DTYPE}")
  fi
  if ! _has_arg "--swap-space" && ! _has_arg "--swap_space"; then
    ASR_MEM_ARGS+=(--swap-space "${ASR_SWAP_SPACE}")
  fi
  # shellcheck disable=SC2206
  ASR_CMD=(
    "${ASR_BIN}"
    "${ASR_MODEL}"
    --host "${ASR_HOST}"
    --port "${ASR_PORT}"
    --gpu-memory-utilization "${ASR_GPU_MEM_UTIL}"
    "${ASR_MEM_ARGS[@]}"
    ${ASR_EXTRA_ARGS}
  )
  # Prepend the ASR venv's bin dir to PATH so that subprocesses spawned by
  # vLLM (e.g. `ninja` for FP8 kernel compilation) can find venv-installed
  # tools — invoking `bin/qwen-asr-serve` directly does not activate the venv.
  spawn "qwen-asr-serve" "${LOG_DIR}/qwen-asr-serve.log" \
    env "PATH=${ASR_VENV}/bin:${PATH}" "VIRTUAL_ENV=${ASR_VENV}" "${ASR_CMD[@]}" \
    || die "qwen-asr-serve failed to start"
  wait_for_http "${ASR_URL_BASE}/models" "qwen-asr-serve" 600 \
    || die "qwen-asr-serve never became healthy"
fi

# ── Phase 3: launch sidecar ──────────────────────────────────────────────────
if (( START_SIDECAR )); then
  if ! port_is_free "${SIDECAR_HOST}" "${SIDECAR_PORT}"; then
    die "sidecar port ${SIDECAR_HOST}:${SIDECAR_PORT} already in use"
  fi
  SIDECAR_ENV=()
  if (( START_ASR )); then
    SIDECAR_ENV+=(
      "ASR_PROVIDER=qwen-local"
      "QWEN_LOCAL_BACKEND=openai"
      "QWEN_LOCAL_MODEL=${ASR_MODEL}"
      "QWEN_LOCAL_OPENAI_BASE_URL=${ASR_URL_BASE}"
    )
  fi
  spawn "xtalk-sidecar" "${LOG_DIR}/xtalk-sidecar.log" \
    env "${SIDECAR_ENV[@]}" "${SIDECAR_VENV}/bin/python" "${SIDECAR_DIR}/app.py" \
    || die "sidecar failed to start"
  wait_for_port "${SIDECAR_HOST}" "${SIDECAR_PORT}" "xtalk-sidecar" 120 \
    || die "sidecar never opened ${SIDECAR_HOST}:${SIDECAR_PORT}"
fi

# ── Phase 4: launch Node bridge ──────────────────────────────────────────────
if (( START_NODE )); then
  if ! have_cmd npm; then
    warn "npm not found; skipping Node bridge"
  else
    if [[ ! -d "${NODE_DIR}/node_modules" ]]; then
      log "installing Node dependencies"
      if [[ -f "${NODE_DIR}/package-lock.json" ]]; then
        ( cd "${NODE_DIR}" && npm ci ) >>"${LOG_DIR}/node-install.log" 2>&1 \
          || die "npm ci failed; see ${LOG_DIR}/node-install.log"
      else
        ( cd "${NODE_DIR}" && npm install ) >>"${LOG_DIR}/node-install.log" 2>&1 \
          || die "npm install failed; see ${LOG_DIR}/node-install.log"
      fi
    fi
    spawn "hermes-bridge" "${LOG_DIR}/hermes-bridge.log" \
      bash -c "cd '${NODE_DIR}' && exec npm start" \
      || die "Node bridge (Hermes) failed to start"
  fi
fi

# ── Steady state ─────────────────────────────────────────────────────────────
echo
ok "all components running. logs in ${LOG_DIR}/"
for ((i=0; i<${#CHILD_PIDS[@]}; i++)); do
  printf '  %s%-20s%s pid=%s  log=%s/%s.log\n' \
    "${C_BLUE}" "${CHILD_NAMES[$i]}" "${C_RESET}" \
    "${CHILD_PIDS[$i]}" "${LOG_DIR}" "${CHILD_NAMES[$i]}"
done
echo
log "press Ctrl-C to stop everything"

while true; do
  for ((i=0; i<${#CHILD_PIDS[@]}; i++)); do
    pid="${CHILD_PIDS[$i]}"
    name="${CHILD_NAMES[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      err "${name} (pid=${pid}) exited unexpectedly"
      err "tail of log:"; tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
      exit 1
    fi
  done
  sleep 2
done
