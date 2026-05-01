#!/usr/bin/env python3
"""
X-Talk Bridge Service – entry point.
Starts the asyncio WebSocket sidecar on 127.0.0.1:7431.
"""
import asyncio
import logging
import sys

from config.config import (
    SIDECAR_HOST,
    SIDECAR_PORT,
    ASR_PROVIDER,
    ASR_MODEL,
    ASR_DEVICE,
    ASR_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    VAD_ENERGY_THRESHOLD,
    VAD_SILENCE_MS,
    VAD_PARTIAL_INTERVAL_MS,
    DASHSCOPE_API_KEY,
    QWEN_ASR_MODEL,
    QWEN_ASR_URL,
    QWEN_ASR_LANGUAGE,
    QWEN_ASR_SAMPLE_RATE,
    QWEN_ASR_TURN_DETECTION_THRESHOLD,
    QWEN_ASR_TURN_DETECTION_SILENCE_MS,
    QWEN_LOCAL_MODEL,
    QWEN_LOCAL_BACKEND,
    QWEN_LOCAL_DEVICE,
    QWEN_LOCAL_DTYPE,
    QWEN_LOCAL_LANGUAGE,
    QWEN_LOCAL_MAX_NEW_TOKENS,
    QWEN_LOCAL_GPU_MEMORY_UTILIZATION,
    QWEN_LOCAL_ATTN_IMPLEMENTATION,
    QWEN_LOCAL_STREAMING_CHUNK_MS,
    QWEN_LOCAL_UNFIXED_CHUNK_NUM,
    QWEN_LOCAL_UNFIXED_TOKEN_NUM,
    QWEN_LOCAL_CHUNK_SIZE_SEC,
    QWEN_LOCAL_PARTIAL_MIN_INTERVAL_MS,
    QWEN_LOCAL_ENERGY_THRESHOLD,
    QWEN_LOCAL_SILENCE_MS,
    QWEN_LOCAL_MIN_SPEECH_MS,
    QWEN_LOCAL_OPENAI_BASE_URL,
    QWEN_LOCAL_OPENAI_API_KEY,
    QWEN_LOCAL_OPENAI_TIMEOUT_S,
    TTS_ENABLED,
    TTS_PROVIDER,
    TTS_MODE,
    TTS_SPEAKER_ID,
    TTS_SPEED,
    COSYVOICE_REPO_DIR,
    TTS_MODEL_DIR,
    TTS_PROMPT_TEXT,
    TTS_PROMPT_WAV,
    TTS_INSTRUCT_TEXT,
    ALIYUN_COSYVOICE_MODEL,
    ALIYUN_COSYVOICE_VOICE,
    ALIYUN_COSYVOICE_AUDIO_FORMAT,
    ALIYUN_COSYVOICE_VOLUME,
    ALIYUN_COSYVOICE_SPEECH_RATE,
    ALIYUN_COSYVOICE_PITCH_RATE,
    ALIYUN_COSYVOICE_INSTRUCTION,
    ALIYUN_COSYVOICE_TIMEOUT_MS,
    OMNIVOICE_MODEL,
    OMNIVOICE_MODEL_DIR,
    OMNIVOICE_DEVICE,
    OMNIVOICE_DTYPE,
    OMNIVOICE_NUM_STEP,
    OMNIVOICE_GUIDANCE_SCALE,
    OMNIVOICE_SPEED,
    OMNIVOICE_REF_AUDIO,
    OMNIVOICE_REF_TEXT,
    OMNIVOICE_INSTRUCT,
    SPEECH_ENHANCER_ENABLED,
    SPEECH_ENHANCER_MODEL,
    SPEECH_ENHANCER_NUM_THREADS,
    SPEECH_ENHANCER_PROVIDER,
)
from xtalk_runtime import (
    CosyVoiceTTS,
    DashScopeCosyVoiceTTS,
    OmniVoiceTTS,
    Qwen3LocalASREngine,
    QwenRealtimeASREngine,
    SpeechEnhancer,
    SpeechEnhancerUnavailableError,
    StubTTS,
    WhisperASREngine,
)
from websocket_server import BridgeWebSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("app")


def _load_whisper_model():
    from faster_whisper import WhisperModel
    log.info(f"Loading Whisper model={ASR_MODEL} device={ASR_DEVICE} compute={ASR_COMPUTE_TYPE}")
    model = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)
    log.info("Whisper model loaded OK.")
    return model


async def _build_asr_engine():
    if ASR_PROVIDER == "qwen-realtime":
        log.info(
            "Configuring Qwen Realtime ASR model=%s url=%s language=%s sample_rate=%s",
            QWEN_ASR_MODEL,
            QWEN_ASR_URL,
            QWEN_ASR_LANGUAGE,
            QWEN_ASR_SAMPLE_RATE,
        )
        return QwenRealtimeASREngine(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_ASR_MODEL,
            url=QWEN_ASR_URL,
            language=QWEN_ASR_LANGUAGE,
            sample_rate=QWEN_ASR_SAMPLE_RATE,
            turn_detection_threshold=QWEN_ASR_TURN_DETECTION_THRESHOLD,
            turn_detection_silence_ms=QWEN_ASR_TURN_DETECTION_SILENCE_MS,
        )

    if ASR_PROVIDER == "qwen-local":
        log.info(
            "Configuring local Qwen3-ASR backend=%s model=%s device=%s dtype=%s language=%s",
            QWEN_LOCAL_BACKEND,
            QWEN_LOCAL_MODEL,
            QWEN_LOCAL_DEVICE,
            QWEN_LOCAL_DTYPE,
            QWEN_LOCAL_LANGUAGE,
        )
        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(
            None,
            lambda: Qwen3LocalASREngine(
                model_path=QWEN_LOCAL_MODEL,
                backend=QWEN_LOCAL_BACKEND,
                device=QWEN_LOCAL_DEVICE,
                dtype=QWEN_LOCAL_DTYPE,
                language=QWEN_LOCAL_LANGUAGE,
                max_new_tokens=QWEN_LOCAL_MAX_NEW_TOKENS,
                gpu_memory_utilization=QWEN_LOCAL_GPU_MEMORY_UTILIZATION,
                attn_implementation=QWEN_LOCAL_ATTN_IMPLEMENTATION or None,
                streaming_chunk_ms=QWEN_LOCAL_STREAMING_CHUNK_MS,
                unfixed_chunk_num=QWEN_LOCAL_UNFIXED_CHUNK_NUM,
                unfixed_token_num=QWEN_LOCAL_UNFIXED_TOKEN_NUM,
                chunk_size_sec=QWEN_LOCAL_CHUNK_SIZE_SEC,
                partial_min_interval_ms=QWEN_LOCAL_PARTIAL_MIN_INTERVAL_MS,
                energy_threshold=QWEN_LOCAL_ENERGY_THRESHOLD,
                silence_limit_ms=QWEN_LOCAL_SILENCE_MS,
                min_speech_ms=QWEN_LOCAL_MIN_SPEECH_MS,
            ),
        )
        if QWEN_LOCAL_BACKEND == "openai":
            engine.configure_openai(
                base_url=QWEN_LOCAL_OPENAI_BASE_URL,
                api_key=QWEN_LOCAL_OPENAI_API_KEY,
                timeout_s=QWEN_LOCAL_OPENAI_TIMEOUT_S,
            )
        return engine

    loop = asyncio.get_event_loop()
    whisper_model = await loop.run_in_executor(None, _load_whisper_model)
    return WhisperASREngine(
        model=whisper_model,
        language=WHISPER_LANGUAGE,
        energy_threshold=VAD_ENERGY_THRESHOLD,
        silence_limit_ms=VAD_SILENCE_MS,
        partial_interval_ms=VAD_PARTIAL_INTERVAL_MS,
    )


def _build_tts_engine():
    if not TTS_ENABLED:
        log.info("TTS disabled; using stub engine")
        return StubTTS()
    if TTS_PROVIDER == "aliyun-cosyvoice":
        log.info(
            "Configuring DashScope CosyVoice TTS model=%s voice=%s format=%s",
            ALIYUN_COSYVOICE_MODEL,
            ALIYUN_COSYVOICE_VOICE or "<unset>",
            ALIYUN_COSYVOICE_AUDIO_FORMAT,
        )
        return DashScopeCosyVoiceTTS(
            api_key=DASHSCOPE_API_KEY,
            model=ALIYUN_COSYVOICE_MODEL,
            voice=ALIYUN_COSYVOICE_VOICE,
            audio_format=ALIYUN_COSYVOICE_AUDIO_FORMAT,
            volume=ALIYUN_COSYVOICE_VOLUME,
            speech_rate=ALIYUN_COSYVOICE_SPEECH_RATE,
            pitch_rate=ALIYUN_COSYVOICE_PITCH_RATE,
            instruction=ALIYUN_COSYVOICE_INSTRUCTION or None,
            timeout_millis=ALIYUN_COSYVOICE_TIMEOUT_MS,
        )
    if TTS_PROVIDER == "omnivoice":
        # Resolve model source: prefer the local directory; fall back to HF download.
        model_source = str(OMNIVOICE_MODEL_DIR) if OMNIVOICE_MODEL_DIR.exists() else OMNIVOICE_MODEL
        log.info(
            "Configuring OmniVoice TTS  source=%s  device=%s  dtype=%s  steps=%d",
            model_source,
            OMNIVOICE_DEVICE,
            OMNIVOICE_DTYPE,
            OMNIVOICE_NUM_STEP,
        )
        return OmniVoiceTTS(
            model_path=model_source,
            device=OMNIVOICE_DEVICE,
            dtype=OMNIVOICE_DTYPE,
            num_step=OMNIVOICE_NUM_STEP,
            guidance_scale=OMNIVOICE_GUIDANCE_SCALE,
            ref_audio=OMNIVOICE_REF_AUDIO or None,
            ref_text=OMNIVOICE_REF_TEXT or None,
            instruct=OMNIVOICE_INSTRUCT or None,
            speed=OMNIVOICE_SPEED,
        )
    if TTS_PROVIDER != "cosyvoice":
        log.warning("Unsupported TTS provider %s; using stub engine", TTS_PROVIDER)
        return StubTTS()
    log.info(
        "Configuring CosyVoice TTS mode=%s speaker=%s model_dir=%s repo_dir=%s",
        TTS_MODE,
        TTS_SPEAKER_ID,
        TTS_MODEL_DIR,
        COSYVOICE_REPO_DIR,
    )
    return CosyVoiceTTS(
        repo_dir=COSYVOICE_REPO_DIR,
        model_dir=TTS_MODEL_DIR,
        speaker_id=TTS_SPEAKER_ID or None,
        inference_mode=TTS_MODE,
        speed=TTS_SPEED,
        prompt_text=TTS_PROMPT_TEXT,
        prompt_wav=TTS_PROMPT_WAV,
        instruct_text=TTS_INSTRUCT_TEXT or None,
    )


def _build_enhancer():
    if not SPEECH_ENHANCER_ENABLED:
        log.info("Fast Enhancer disabled (SPEECH_ENHANCER_ENABLED=0)")
        return None
    try:
        return SpeechEnhancer(
            model_path=SPEECH_ENHANCER_MODEL,
            num_threads=SPEECH_ENHANCER_NUM_THREADS,
            provider=SPEECH_ENHANCER_PROVIDER,
        )
    except SpeechEnhancerUnavailableError as exc:
        log.warning(
            "Fast Enhancer unavailable; running without noise denoising. "
            "Reason: %s",
            exc,
        )
        return None


async def main():
    log.info("=" * 60)
    log.info("X-Talk Bridge Service starting up")
    log.info(f"  Sidecar address : ws://{SIDECAR_HOST}:{SIDECAR_PORT}")
    log.info(f"  ASR provider    : {ASR_PROVIDER}")
    log.info(f"  Language        : {WHISPER_LANGUAGE}")
    log.info(f"  TTS enabled     : {TTS_ENABLED} provider={TTS_PROVIDER} mode={TTS_MODE}")
    log.info(f"  Fast Enhancer   : {SPEECH_ENHANCER_ENABLED} model={SPEECH_ENHANCER_MODEL}")
    log.info("=" * 60)

    asr_engine = await _build_asr_engine()
    tts_engine = _build_tts_engine()
    enhancer = _build_enhancer()

    server = BridgeWebSocketServer(
        host=SIDECAR_HOST,
        port=SIDECAR_PORT,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        enhancer=enhancer,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
