"""
xtalk_runtime.py - ASR/TTS runtime implementations for local and official providers.
"""
import asyncio
import base64
import io
import logging
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Hz - must match browser capture

AsrPartialCallback = Callable[[str], Awaitable[None]]
AsrFinalCallback = Callable[[str, dict | None], Awaitable[None]]
AsrErrorCallback = Callable[[str], Awaitable[None]]
AsrSpeechStartedCallback = Callable[[], Awaitable[None]]


class TTSUnavailableError(RuntimeError):
    pass


class ASRUnavailableError(RuntimeError):
    pass


class SpeechEnhancerUnavailableError(RuntimeError):
    pass


class SpeechEnhancer:
    """Lightweight speech denoiser based on k2-fsa sherpa-onnx GTCRN ("Fast Enhancer").

    Strips stationary background noise (HVAC, fan, keyboard hiss, distant TV,
    speaker bleed during AI playback) from incoming 16 kHz mono PCM-16 audio.
    The model is < 1 MB and runs at RTF ~ 0.07 on a single CPU thread, so it
    can sit on the realtime audio path with negligible added latency
    (~3 ms per 60 ms chunk).

    The denoiser is fed every microphone chunk *before* it is forwarded to the
    ASR or used to compute barge-in energy. After denoising, ambient noise
    collapses to ~0 RMS, which is the single biggest reason it fixes the
    "AI gets interrupted by noise mid-sentence" failure mode.

    Voice-only spec: input is 16-bit signed-integer mono PCM at 16 kHz; output
    is the same format and length, ready to be fed to either ASR or RMS
    computation.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: str | Path,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        self._model_path = str(model_path)
        self._num_threads = max(1, int(num_threads))
        self._provider = provider
        self._denoiser = None
        self._lock = None
        self._init_denoiser()

    def _init_denoiser(self) -> None:
        if not Path(self._model_path).exists():
            raise SpeechEnhancerUnavailableError(
                f"Fast Enhancer model not found at {self._model_path}; "
                "run scripts/bootstrap_fast_enhancer.py first",
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SpeechEnhancerUnavailableError(
                "sherpa-onnx is not installed; run: pip install sherpa-onnx",
            ) from exc

        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=self._model_path,
                ),
                num_threads=self._num_threads,
                debug=False,
                provider=self._provider,
            )
        )
        if not config.validate():
            raise SpeechEnhancerUnavailableError(
                f"Fast Enhancer config invalid for model {self._model_path}",
            )

        log.info(
            "Loading Fast Enhancer (sherpa-onnx GTCRN) model=%s provider=%s threads=%d",
            self._model_path,
            self._provider,
            self._num_threads,
        )
        t0 = time.perf_counter()
        self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        # Single-instance ONNX inference is not necessarily thread-safe; serialise
        # access from the asyncio thread pool with a threading.Lock.
        import threading
        self._lock = threading.Lock()
        log.info("Fast Enhancer ready in %.0f ms", (time.perf_counter() - t0) * 1000)

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def enhance_pcm16(self, pcm_bytes: bytes) -> bytes:
        """Denoise a chunk of 16 kHz mono PCM-16 audio.

        Returns a bytes object of the same length as the input. On any internal
        failure the original bytes are returned unchanged so the audio path
        never goes silent.
        """
        if self._denoiser is None or not pcm_bytes:
            return pcm_bytes
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if samples.size == 0:
                return pcm_bytes
            audio_f32 = samples.astype(np.float32) / 32768.0
            with self._lock:
                result = self._denoiser.run(audio_f32, self.SAMPLE_RATE)
            denoised = np.asarray(result.samples, dtype=np.float32)
            # GTCRN may return a slightly shorter/longer tail (FFT framing).
            # Pad/truncate to match the input length so downstream code can rely
            # on chunk-size invariants.
            if denoised.size != samples.size:
                if denoised.size > samples.size:
                    denoised = denoised[: samples.size]
                else:
                    pad = np.zeros(samples.size - denoised.size, dtype=np.float32)
                    denoised = np.concatenate([denoised, pad])
            denoised = np.clip(denoised, -1.0, 1.0)
            pcm16 = (denoised * 32767.0).astype(np.int16)
            return pcm16.tobytes()
        except Exception:
            log.exception("Fast Enhancer failed; falling back to raw audio")
            return pcm_bytes


def time_ms() -> int:
    return time.time_ns() // 1_000_000


def _log_background_result(future) -> None:
    try:
        future.result()
    except Exception:
        log.exception("Background callback failed")


def _schedule_async(loop: asyncio.AbstractEventLoop, coro: Awaitable[None]) -> None:
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    future.add_done_callback(_log_background_result)


def _wav_bytes_from_float32(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _wav_bytes_from_pcm16(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _sample_rate_from_audio_format(format_name: str, default: int = 24000) -> int:
    match = re.search(r"_(\d+)HZ_", format_name.upper())
    if not match:
        return default
    return int(match.group(1))


class StubTTS:
    sample_rate = 24000
    mime_type = "audio/wav"

    def synthesize(self, text: str) -> bytes:
        log.debug("[StubTTS] synthesize: %r", text)
        return b""


class DashScopeCosyVoiceTTS:
    mime_type = "audio/wav"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        audio_format: str = "PCM_24000HZ_MONO_16BIT",
        volume: int = 50,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
        instruction: str | None = None,
        additional_params: dict | None = None,
        timeout_millis: int = 30000,
    ):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._model = model
        self._voice = voice
        self._audio_format_name = audio_format
        self._volume = volume
        self._speech_rate = speech_rate
        self._pitch_rate = pitch_rate
        self._instruction = instruction
        self._additional_params = additional_params or {}
        self._timeout_millis = timeout_millis
        self.sample_rate = _sample_rate_from_audio_format(audio_format)

    def synthesize(self, text: str) -> bytes:
        normalized = text.strip()
        if not normalized:
            return b""
        if not self._api_key:
            raise TTSUnavailableError("DASHSCOPE_API_KEY is required for Aliyun CosyVoice TTS")
        if not self._voice:
            raise TTSUnavailableError(
                "ALIYUN_COSYVOICE_VOICE is required for cosyvoice-v3.5-flash; it must be a clone/design voice ID",
            )

        try:
            import dashscope
            from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise TTSUnavailableError(
                "DashScope SDK is unavailable; install dashscope>=1.25.6",
            ) from exc

        dashscope.api_key = self._api_key
        format_enum = getattr(AudioFormat, self._audio_format_name, None)
        if format_enum is None:
            raise TTSUnavailableError(f"Unsupported audio format: {self._audio_format_name}")

        synthesizer_kwargs = {
            "model": self._model,
            "voice": self._voice,
            "format": format_enum,
            "volume": self._volume,
            "speech_rate": self._speech_rate,
            "pitch_rate": self._pitch_rate,
        }
        if self._instruction:
            synthesizer_kwargs["instruction"] = self._instruction
        if self._additional_params:
            synthesizer_kwargs["additional_params"] = self._additional_params

        try:
            synthesizer = SpeechSynthesizer(**synthesizer_kwargs)
            audio_bytes = synthesizer.call(normalized, timeout_millis=self._timeout_millis)
        except Exception as exc:  # pragma: no cover - depends on remote service
            raise TTSUnavailableError(f"DashScope CosyVoice request failed: {exc}") from exc

        if not audio_bytes:
            return b""
        if self._audio_format_name.upper().startswith("PCM_"):
            return _wav_bytes_from_pcm16(audio_bytes, self.sample_rate)
        if self._audio_format_name.upper().startswith("WAV_"):
            return audio_bytes

        raise TTSUnavailableError(
            f"Unsupported bridge playback format {self._audio_format_name}; use PCM_* or WAV_*",
        )


class CosyVoiceTTS:
    mime_type = "audio/wav"

    def __init__(
        self,
        repo_dir: str | Path,
        model_dir: str | Path,
        speaker_id: str | None = None,
        inference_mode: str = "sft",
        speed: float = 1.0,
        prompt_text: str | None = None,
        prompt_wav: str | Path | None = None,
        instruct_text: str | None = None,
    ):
        self._repo_dir = Path(repo_dir)
        self._model_dir = Path(model_dir)
        self._speaker_id = speaker_id
        self._inference_mode = inference_mode
        self._speed = speed
        self._prompt_text = prompt_text
        self._prompt_wav = Path(prompt_wav) if prompt_wav else None
        self._instruct_text = instruct_text
        self._model = None
        self.sample_rate = 24000
        self._available_speakers: list[str] | None = None

    def synthesize(self, text: str) -> bytes:
        normalized = text.strip()
        if not normalized:
            return b""

        model = self._ensure_model()
        chunks: list[np.ndarray] = []
        for result in self._run_inference(model, normalized):
            audio = result.get("tts_speech")
            audio_np = self._to_numpy(audio)
            if audio_np.size > 0:
                chunks.append(audio_np)

        if not chunks:
            return b""

        merged = np.concatenate(chunks)
        return _wav_bytes_from_float32(merged, self.sample_rate)

    def _run_inference(self, model, text: str):
        if self._inference_mode == "sft":
            speaker_id = self._resolve_speaker_id(model)
            return model.inference_sft(text, speaker_id, stream=False, speed=self._speed)
        if self._inference_mode == "zero_shot":
            prompt_wav = self._require_prompt_wav()
            if not self._prompt_text:
                raise TTSUnavailableError("TTS_PROMPT_TEXT is required when TTS_MODE=zero_shot")
            return model.inference_zero_shot(
                text,
                self._prompt_text,
                str(prompt_wav),
                stream=False,
                speed=self._speed,
            )
        if self._inference_mode == "instruct2":
            prompt_wav = self._require_prompt_wav()
            if not self._instruct_text:
                raise TTSUnavailableError("TTS_INSTRUCT_TEXT is required when TTS_MODE=instruct2")
            return model.inference_instruct2(
                text,
                self._instruct_text,
                str(prompt_wav),
                stream=False,
                speed=self._speed,
            )
        raise TTSUnavailableError(
            f"Unsupported TTS_MODE={self._inference_mode!r}; use 'zero_shot', 'instruct2', or 'sft'",
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        if not self._repo_dir.exists():
            raise TTSUnavailableError(
                f"CosyVoice repo not found at {self._repo_dir}; run scripts/bootstrap_cosyvoice.py first",
            )
        if not self._model_dir.exists():
            raise TTSUnavailableError(
                f"CosyVoice model not found at {self._model_dir}; run scripts/bootstrap_cosyvoice.py first",
            )

        repo_str = str(self._repo_dir)
        matcha_dir = self._repo_dir / "third_party" / "Matcha-TTS"
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        if matcha_dir.exists() and str(matcha_dir) not in sys.path:
            sys.path.insert(0, str(matcha_dir))

        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore[reportMissingImports]
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise TTSUnavailableError(
                "Failed to import CosyVoice runtime; install CosyVoice dependencies first",
            ) from exc

        log.info("Loading CosyVoice model from %s", self._model_dir)
        self._model = AutoModel(model_dir=str(self._model_dir))
        self.sample_rate = int(getattr(self._model, "sample_rate", self.sample_rate))
        self._available_speakers = list(self._model.list_available_spks())
        log.info(
            "CosyVoice ready sample_rate=%s speakers=%s",
            self.sample_rate,
            ", ".join(self._available_speakers[:8]) if self._available_speakers else "<none>",
        )
        return self._model

    def _resolve_speaker_id(self, model) -> str:
        speakers = self._available_speakers
        if speakers is None:
            speakers = list(model.list_available_spks())
            self._available_speakers = speakers
        if not speakers:
            raise TTSUnavailableError("CosyVoice did not expose any SFT speakers")
        if self._speaker_id and self._speaker_id in speakers:
            return self._speaker_id
        if self._speaker_id and self._speaker_id not in speakers:
            log.warning(
                "Configured TTS speaker %r not found; falling back to %r",
                self._speaker_id,
                speakers[0],
            )
        self._speaker_id = speakers[0]
        return self._speaker_id

    def _require_prompt_wav(self) -> Path:
        if not self._prompt_wav or not self._prompt_wav.exists():
            raise TTSUnavailableError(
                f"Prompt wav not found at {self._prompt_wav}; set TTS_PROMPT_WAV to a valid file",
            )
        return self._prompt_wav

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1)


class OmniVoiceTTS:
    """High-performance local TTS backed by the OmniVoice diffusion-language model.

    The model is loaded eagerly at construction time so that the first
    ``synthesize()`` call incurs no model-loading overhead.  On a warm GPU with
    ``num_step=8`` the time-to-first-audio is roughly 250 ms (RTF ≈ 0.05).

    Voice modes (in priority order):
      1. Voice cloning  – provide ``ref_audio`` + ``ref_text``.
      2. Voice design   – provide ``instruct`` (e.g. ``"female, low pitch"``).
      3. Auto voice     – leave everything empty; OmniVoice picks a voice.
    """

    mime_type = "audio/wav"
    sample_rate = 24000

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        dtype: str = "float16",
        num_step: int = 8,
        guidance_scale: float = 2.0,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        speed: float = 1.0,
    ):
        self._ref_audio = str(ref_audio) if ref_audio else None
        self._ref_text = ref_text or None
        # instruct is only used when no ref_audio is given
        self._instruct = (instruct or None) if not ref_audio else None
        self._speed = speed
        self._num_step = num_step

        try:
            import torch
            from omnivoice import OmniVoice, OmniVoiceGenerationConfig
        except ImportError as exc:
            raise TTSUnavailableError(
                "omnivoice is not installed; run: pip install omnivoice"
            ) from exc

        _dtype = torch.float16 if dtype == "float16" else torch.float32
        # Skip loading the built-in Whisper ASR when ref_text is already known,
        # saving ~500 MB of VRAM and several seconds of startup time.
        _load_asr = bool(self._ref_audio) and (self._ref_text is None)

        model_key = str(model_path)
        log.info(
            "Loading OmniVoice model  path=%s  device=%s  dtype=%s  steps=%d  load_asr=%s",
            model_key,
            device,
            dtype,
            num_step,
            _load_asr,
        )
        t0 = time.perf_counter()
        self._model = OmniVoice.from_pretrained(
            model_key,
            device_map=device,
            dtype=_dtype,
            load_asr=_load_asr,
        )
        log.info("OmniVoice model ready in %.1f s", time.perf_counter() - t0)

        # Pre-build generation config once; reused on every synthesize() call.
        self._gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=True,
            # position_temperature=5.0 preserves the natural rhythm of speech.
            position_temperature=5.0,
            # class_temperature=0.0 → greedy decoding: lowest latency, most stable.
            class_temperature=0.0,
            # Skip pre/post-processing hooks that are only needed for batch/demo use.
            preprocess_prompt=False,
            postprocess_output=False,
        )

        # Pre-compute the voice-clone prompt once at startup so that every
        # synthesize() call can pass a pre-built prompt object instead of
        # re-encoding the reference audio on every request (~10–30 ms saved).
        # Falls back to None when running in voice-design or auto-voice mode.
        self._voice_clone_prompt = None
        if self._ref_audio:
            ref_path = Path(self._ref_audio)
            if ref_path.exists():
                try:
                    t1 = time.perf_counter()
                    self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                        ref_audio=self._ref_audio,
                        ref_text=self._ref_text,
                    )
                    log.info(
                        "OmniVoice voice-clone prompt built in %.2f s",
                        time.perf_counter() - t1,
                    )
                except Exception:
                    log.warning(
                        "OmniVoice create_voice_clone_prompt failed; "
                        "will pass ref_audio path on every call instead",
                        exc_info=True,
                    )
            else:
                log.warning("OmniVoice ref_audio not found at %s; voice cloning disabled", self._ref_audio)

    def synthesize(self, text: str) -> bytes:
        normalized = text.strip()
        if not normalized:
            return b""

        t0 = time.perf_counter()
        try:
            if self._voice_clone_prompt is not None:
                # Fast path: pre-built prompt object, no per-call ref-audio encoding.
                audio_list = self._model.generate(
                    text=normalized,
                    voice_clone_prompt=self._voice_clone_prompt,
                    speed=self._speed,
                    generation_config=self._gen_config,
                )
            else:
                audio_list = self._model.generate(
                    text=normalized,
                    ref_audio=self._ref_audio,
                    ref_text=self._ref_text,
                    instruct=self._instruct,
                    speed=self._speed,
                    generation_config=self._gen_config,
                )
        except Exception as exc:
            raise TTSUnavailableError(f"OmniVoice synthesis failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not audio_list:
            return b""

        audio_np = np.asarray(audio_list[0], dtype=np.float32)
        audio_duration_s = len(audio_np) / self.sample_rate
        rtf = (elapsed_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0
        log.info(
            "[OmniVoice] chars=%d  ttfa=%.0f ms  dur=%.2f s  RTF=%.3f",
            len(normalized),
            elapsed_ms,
            audio_duration_s,
            rtf,
        )
        return _wav_bytes_from_float32(audio_np, self.sample_rate)


class WhisperASREngine:
    def __init__(
        self,
        model,
        language: str = "zh",
        energy_threshold: float = 200.0,
        silence_limit_ms: int = 800,
        partial_interval_ms: int = 1500,
    ):
        self._model = model
        self._language = language
        self._energy_threshold = energy_threshold
        self._silence_limit_ms = silence_limit_ms
        self._partial_interval_ms = partial_interval_ms

    def create_session(
        self,
        *,
        session_id: str,
        turn_id: str,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
    ):
        return WhisperASRSession(
            model=self._model,
            language=self._language,
            energy_threshold=self._energy_threshold,
            silence_limit_ms=self._silence_limit_ms,
            partial_interval_ms=self._partial_interval_ms,
            on_speech_started=on_speech_started,
            on_partial=on_partial,
            on_final=on_final,
            on_error=on_error,
            session_id=session_id,
            turn_id=turn_id,
        )


class WhisperASRSession:
    def __init__(
        self,
        model,
        language: str,
        energy_threshold: float,
        silence_limit_ms: int,
        partial_interval_ms: int,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
        session_id: str,
        turn_id: str,
    ):
        self._model = model
        self._language = language
        self._energy_threshold = energy_threshold
        self._silence_limit = silence_limit_ms / 1000.0
        self._partial_interval = partial_interval_ms / 1000.0
        self._on_speech_started = on_speech_started
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._session_id = session_id
        self._turn_id = turn_id
        self._buffer: list[np.ndarray] = []
        self._silent_duration = 0.0
        self._time_since_partial = 0.0
        self._is_speaking = False

    async def start(self) -> None:
        return

    async def send_audio(self, pcm_bytes: bytes) -> None:
        try:
            result, chunk_received_at_ms = await self._feed(pcm_bytes)
            if result and result.get("kind") == "partial":
                audio = self._get_buffer_copy()
                text = await asyncio.to_thread(self._transcribe, audio)
                if text:
                    await self._on_partial(text)
            elif result and result.get("kind") == "final":
                await self._emit_final(result, chunk_received_at_ms)
        except Exception as exc:  # pragma: no cover - runtime dependent
            log.exception("Whisper ASR send_audio failed session=%s turn=%s", self._session_id, self._turn_id)
            await self._on_error(f"ASR failed: {exc}")

    async def finish(self) -> None:
        try:
            audio = self._get_buffer_and_reset()
            if len(audio) < SAMPLE_RATE * 0.3:
                self._reset_state()
                return
            started_at_ms = time_ms()
            text = await asyncio.to_thread(self._transcribe, audio)
            finished_at_ms = time_ms()
            self._reset_state()
            if text:
                timing = {
                    "speechEndedAtMs": started_at_ms,
                    "sttLatencyMs": finished_at_ms - started_at_ms,
                    "endpointWaitMs": 0,
                    "transcribeDurationMs": finished_at_ms - started_at_ms,
                }
                await self._on_final(text, timing)
        except Exception as exc:  # pragma: no cover - runtime dependent
            log.exception("Whisper ASR finish failed session=%s turn=%s", self._session_id, self._turn_id)
            await self._on_error(f"ASR finish failed: {exc}")

    async def cancel(self) -> None:
        self._reset_state()

    async def _feed(self, pcm_bytes: bytes) -> tuple[dict | None, int]:
        if not pcm_bytes:
            return None, time_ms()

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        chunk_duration = len(samples) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(samples ** 2)))
        chunk_received_at_ms = time_ms()

        if rms > self._energy_threshold:
            if not self._is_speaking:
                await self._on_speech_started()
            self._is_speaking = True
            self._silent_duration = 0.0
            self._buffer.append(samples)
            self._time_since_partial += chunk_duration
            if self._time_since_partial >= self._partial_interval:
                self._time_since_partial = 0.0
                return {"kind": "partial"}, chunk_received_at_ms
        elif self._is_speaking:
            self._silent_duration += chunk_duration
            self._buffer.append(samples)
            if self._silent_duration >= self._silence_limit:
                speech_end_offset_ms = int(self._silent_duration * 1000)
                self._is_speaking = False
                self._silent_duration = 0.0
                self._time_since_partial = 0.0
                return {
                    "kind": "final",
                    "speech_end_offset_ms": speech_end_offset_ms,
                }, chunk_received_at_ms

        return None, chunk_received_at_ms

    async def _emit_final(self, result: dict, chunk_received_at_ms: int) -> None:
        audio = self._get_buffer_and_reset()
        speech_end_offset_ms = int(result.get("speech_end_offset_ms", 0))
        speech_ended_at_ms = chunk_received_at_ms - speech_end_offset_ms
        transcribe_started_at_ms = time_ms()
        text = await asyncio.to_thread(self._transcribe, audio)
        transcribe_finished_at_ms = time_ms()
        self._reset_state()
        if text:
            await self._on_final(
                text,
                {
                    "speechEndedAtMs": speech_ended_at_ms,
                    "sttLatencyMs": transcribe_finished_at_ms - speech_ended_at_ms,
                    "endpointWaitMs": transcribe_started_at_ms - speech_ended_at_ms,
                    "transcribeDurationMs": transcribe_finished_at_ms - transcribe_started_at_ms,
                },
            )

    def _get_buffer_copy(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buffer)

    def _get_buffer_and_reset(self) -> np.ndarray:
        audio = self._get_buffer_copy()
        self._buffer = []
        return audio

    def _reset_state(self) -> None:
        self._buffer = []
        self._silent_duration = 0.0
        self._time_since_partial = 0.0
        self._is_speaking = False

    def _transcribe(self, audio_f32: np.ndarray) -> str:
        if len(audio_f32) < SAMPLE_RATE * 0.3:
            return ""
        audio_norm = audio_f32 / 32768.0
        segments, _info = self._model.transcribe(
            audio_norm,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()


class QwenRealtimeASREngine:
    def __init__(
        self,
        api_key: str,
        model: str,
        url: str,
        language: str,
        sample_rate: int,
        turn_detection_threshold: float,
        turn_detection_silence_ms: int,
    ):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._model = model
        self._url = url
        self._language = language
        self._sample_rate = sample_rate
        self._turn_detection_threshold = turn_detection_threshold
        self._turn_detection_silence_ms = turn_detection_silence_ms

    def create_session(
        self,
        *,
        session_id: str,
        turn_id: str,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
    ):
        return QwenRealtimeASRSession(
            api_key=self._api_key,
            model=self._model,
            url=self._url,
            language=self._language,
            sample_rate=self._sample_rate,
            turn_detection_threshold=self._turn_detection_threshold,
            turn_detection_silence_ms=self._turn_detection_silence_ms,
            on_speech_started=on_speech_started,
            on_partial=on_partial,
            on_final=on_final,
            on_error=on_error,
            session_id=session_id,
            turn_id=turn_id,
        )


class QwenRealtimeASRSession:
    def __init__(
        self,
        api_key: str,
        model: str,
        url: str,
        language: str,
        sample_rate: int,
        turn_detection_threshold: float,
        turn_detection_silence_ms: int,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
        session_id: str,
        turn_id: str,
    ):
        self._api_key = api_key
        self._model = model
        self._url = url
        self._language = language
        self._sample_rate = sample_rate
        self._turn_detection_threshold = turn_detection_threshold
        self._turn_detection_silence_ms = turn_detection_silence_ms
        self._on_speech_started = on_speech_started
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._session_id = session_id
        self._turn_id = turn_id
        self._loop = asyncio.get_running_loop()
        self._conversation = None
        self._started = False
        self._closed = False
        self._finished = False
        self._speech_started = False
        self._last_partial_preview = ""
        self._speech_stopped_at_ms: int | None = None
        self._last_audio_sent_at_ms: int | None = None

    async def start(self) -> None:
        if self._started:
            return
        if not self._api_key:
            raise ASRUnavailableError("DASHSCOPE_API_KEY is required for Qwen realtime ASR")
        await asyncio.to_thread(self._connect)
        self._started = True

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        await self.start()
        if self._closed or self._conversation is None:
            return
        self._last_audio_sent_at_ms = time_ms()
        audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
        await asyncio.to_thread(self._conversation.append_audio, audio_b64)

    async def finish(self) -> None:
        await self.start()
        if self._closed or self._finished or self._conversation is None:
            return
        self._finished = True
        try:
            await asyncio.to_thread(self._conversation.end_session)
        finally:
            await asyncio.to_thread(self._conversation.close)
            self._closed = True

    async def cancel(self) -> None:
        if self._closed or self._conversation is None:
            return
        self._closed = True
        await asyncio.to_thread(self._conversation.close)

    def _connect(self) -> None:
        try:
            import dashscope
            from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeCallback, OmniRealtimeConversation
            from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ASRUnavailableError(
                "DashScope SDK is unavailable; install dashscope>=1.25.6",
            ) from exc

        dashscope.api_key = self._api_key
        owner = self

        class Callback(OmniRealtimeCallback):
            def on_open(self):
                log.info("Qwen realtime ASR opened session=%s turn=%s", owner._session_id, owner._turn_id)

            def on_event(self, message: dict):
                owner._handle_event(message)

            def on_close(self, close_status_code, close_msg):
                if owner._closed:
                    return
                if close_status_code not in (1000, None):
                    _schedule_async(
                        owner._loop,
                        owner._on_error(
                            f"Qwen realtime ASR closed unexpectedly: code={close_status_code} msg={close_msg}",
                        ),
                    )

        callback = Callback()
        self._conversation = OmniRealtimeConversation(
            model=self._model,
            url=self._url,
            callback=callback,
        )
        self._conversation.connect()
        transcription_params = TranscriptionParams(
            language=self._language,
            sample_rate=self._sample_rate,
            input_audio_format="pcm",
        )
        self._conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            enable_turn_detection=True,
            turn_detection_type="server_vad",
            turn_detection_threshold=self._turn_detection_threshold,
            turn_detection_silence_duration_ms=self._turn_detection_silence_ms,
            transcription_params=transcription_params,
        )

    def _handle_event(self, message: dict) -> None:
        if self._closed:
            return
        event_type = message.get("type")
        if event_type == "input_audio_buffer.speech_started":
            if not self._speech_started:
                self._speech_started = True
                _schedule_async(self._loop, self._on_speech_started())
            return

        if event_type == "conversation.item.input_audio_transcription.text":
            preview = f"{message.get('text', '')}{message.get('stash', '')}".strip()
            if preview and preview != self._last_partial_preview:
                self._last_partial_preview = preview
                _schedule_async(self._loop, self._on_partial(preview))
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(message.get("transcript") or "").strip()
            if transcript:
                _schedule_async(self._loop, self._on_final(transcript, self._build_timing()))
            return

        if event_type == "conversation.item.input_audio_transcription.failed":
            error = message.get("error") or {}
            error_message = error.get("message") if isinstance(error, dict) else None
            _schedule_async(
                self._loop,
                self._on_error(error_message or "Qwen realtime ASR transcription failed"),
            )
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self._speech_stopped_at_ms = time_ms()
            return

        if event_type == "error":
            error = message.get("error") or {}
            error_message = error.get("message") if isinstance(error, dict) else None
            _schedule_async(
                self._loop,
                self._on_error(error_message or "Qwen realtime ASR returned an error"),
            )

    def _build_timing(self) -> dict:
        finished_at_ms = time_ms()
        speech_ended_at_ms = self._speech_stopped_at_ms or self._last_audio_sent_at_ms or finished_at_ms
        latency_ms = max(0, finished_at_ms - speech_ended_at_ms)
        return {
            "speechEndedAtMs": speech_ended_at_ms,
            "sttLatencyMs": latency_ms,
            "endpointWaitMs": latency_ms,
            "transcribeDurationMs": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Local Qwen3-ASR (open-source) — true streaming via vLLM backend, with a
# transformers backend as a non-streaming fallback. Loaded once at startup;
# every turn gets its own per-session streaming state.
# ─────────────────────────────────────────────────────────────────────────────


_LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "nl": "Dutch",
    "tr": "Turkish",
    "hi": "Hindi",
    "pl": "Polish",
}


def _normalize_qwen_language(language: str | None) -> str | None:
    if not language:
        return None
    code = language.strip().lower()
    if not code:
        return None
    return _LANGUAGE_CODE_TO_NAME.get(code, language)


_QWEN_ASR_TEXT_TAG_RE = re.compile(r"<asr_text>", re.IGNORECASE)
_QWEN_LANGUAGE_PREFIX_RE = re.compile(
    r"^\s*language\s+(?:none|"
    + "|".join(sorted(set(_LANGUAGE_CODE_TO_NAME.values())))
    + r")(?=\s|<|$|[^\x00-\x7f])",
    re.IGNORECASE,
)


def _clean_qwen_asr_text(raw: str | None) -> str:
    """Normalize Qwen3-ASR raw decoder text to transcript-only text."""
    if raw is None:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    # Older qwen-asr/vLLM paths may return Whisper-style language sentinels.
    text = re.sub(r"^\s*<\|[^|]+\|>\s*", "", text).strip()

    # Qwen3-ASR can emit metadata in the form:
    #   language Chinese<asr_text>你好
    #   language None<asr_text>你好
    # The model's own parser keeps the text after the tag; mirror that here so
    # split-process mode never forwards metadata into the OpenClaw turn.
    match = _QWEN_ASR_TEXT_TAG_RE.search(text)
    if match:
        meta = text[: match.start()]
        body = text[match.end() :].strip()
        if _QWEN_LANGUAGE_PREFIX_RE.search(meta):
            return body
        return body or meta.strip()

    # Be defensive for malformed outputs missing the tag but still starting
    # with the language preamble.
    text = _QWEN_LANGUAGE_PREFIX_RE.sub("", text, count=1).strip()
    return text


class Qwen3LocalASREngine:
    """Local open-source Qwen3-ASR engine (vLLM streaming or transformers).

    The model is loaded once at startup and reused across turns. Each call to
    ``create_session`` returns a fresh per-turn session that owns its own
    streaming state, so concurrent turns from different browser tabs do not
    interfere with each other.

    Backends:
      * ``vllm`` (recommended for single-venv deploys): true streaming inference
        via ``Qwen3ASRModel.LLM(...)`` and ``streaming_transcribe(...)``.
        Requires ``pip install qwen-asr[vllm]`` and a CUDA GPU. Conflicts
        with ``omnivoice`` because of incompatible transformers pins.
      * ``transformers``: non-streaming fallback using ``model.transcribe(...)``
        on the full VAD-segmented utterance. Same dependency conflict as vllm.
      * ``openai``: split-process mode — talks to an external
        ``qwen-asr-serve`` instance over its OpenAI-compatible HTTP API.
        This sidecar venv stays free of qwen-asr/vllm so it can keep using
        omnivoice. No streaming partials; one final per VAD utterance.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: str | None = "zh",
        max_new_tokens: int = 64,
        gpu_memory_utilization: float = 0.5,
        attn_implementation: str | None = None,
        # Streaming policy
        streaming_chunk_ms: int = 480,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        partial_min_interval_ms: int = 200,
        # VAD endpointing
        energy_threshold: float = 200.0,
        silence_limit_ms: int = 600,
        min_speech_ms: int = 200,
    ):
        self._model_path = model_path
        self._backend = backend.lower().strip()
        self._device = device
        self._dtype = dtype
        self._language = _normalize_qwen_language(language)
        self._max_new_tokens = max_new_tokens
        self._gpu_memory_utilization = gpu_memory_utilization
        self._attn_implementation = attn_implementation or None
        self._streaming_chunk_ms = max(80, int(streaming_chunk_ms))
        self._unfixed_chunk_num = unfixed_chunk_num
        self._unfixed_token_num = unfixed_token_num
        self._chunk_size_sec = chunk_size_sec
        self._partial_min_interval_ms = partial_min_interval_ms
        self._energy_threshold = energy_threshold
        self._silence_limit_ms = silence_limit_ms
        self._min_speech_ms = min_speech_ms

        if self._backend not in {"vllm", "transformers", "openai"}:
            raise ASRUnavailableError(
                f"Unsupported QWEN_LOCAL_BACKEND={backend!r}; use 'vllm', 'transformers', or 'openai'",
            )

        self._model = None
        self._model_lock = None  # type: ignore[assignment]
        # openai backend extras (set via configure_openai())
        self._openai_base_url: str | None = None
        self._openai_api_key: str = "EMPTY"
        self._openai_timeout_s: float = 30.0
        if self._backend != "openai":
            self._init_model()
        else:
            import threading
            self._model_lock = threading.Lock()
            log.info(
                "Configured local Qwen3-ASR in split-process (openai) mode  model=%s",
                model_path,
            )

    def configure_openai(self, *, base_url: str, api_key: str = "EMPTY", timeout_s: float = 30.0) -> None:
        if self._backend != "openai":
            return
        self._openai_base_url = base_url.rstrip("/")
        self._openai_api_key = api_key or "EMPTY"
        self._openai_timeout_s = timeout_s
        log.info(
            "Local Qwen3-ASR openai backend  url=%s  model=%s  timeout=%.1fs",
            self._openai_base_url,
            self._model_path,
            timeout_s,
        )

    def _init_model(self) -> None:
        try:
            import torch  # type: ignore[reportMissingImports]
            from qwen_asr import Qwen3ASRModel  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ASRUnavailableError(
                "qwen-asr is not installed; run: pip install -U qwen-asr"
                + (" [vllm]" if self._backend == "vllm" else ""),
            ) from exc

        torch_dtype = getattr(torch, self._dtype, None)
        if torch_dtype is None:
            raise ASRUnavailableError(f"Unknown torch dtype: {self._dtype!r}")

        log.info(
            "Loading local Qwen3-ASR  backend=%s  model=%s  device=%s  dtype=%s",
            self._backend,
            self._model_path,
            self._device,
            self._dtype,
        )
        t0 = time.perf_counter()
        if self._backend == "vllm":
            kwargs = dict(
                model=self._model_path,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_new_tokens=self._max_new_tokens,
            )
            try:
                self._model = Qwen3ASRModel.LLM(**kwargs)
            except Exception as exc:
                raise ASRUnavailableError(
                    f"Failed to initialise Qwen3-ASR vLLM backend: {exc}",
                ) from exc
        else:
            tf_kwargs = dict(
                dtype=torch_dtype,
                device_map=self._device,
                max_inference_batch_size=1,
                max_new_tokens=self._max_new_tokens,
            )
            if self._attn_implementation:
                tf_kwargs["attn_implementation"] = self._attn_implementation
            try:
                self._model = Qwen3ASRModel.from_pretrained(self._model_path, **tf_kwargs)
            except Exception as exc:
                raise ASRUnavailableError(
                    f"Failed to load Qwen3-ASR transformers backend: {exc}",
                ) from exc
        import threading
        # Serialise model access; Qwen3ASRModel is not guaranteed thread-safe
        # for concurrent streaming_transcribe calls across sessions on the same
        # state, and the underlying engine batches internally anyway.
        self._model_lock = threading.Lock()
        log.info(
            "Local Qwen3-ASR ready in %.1f s  (chunk=%dms  silence=%dms  language=%s)",
            time.perf_counter() - t0,
            self._streaming_chunk_ms,
            self._silence_limit_ms,
            self._language or "<auto>",
        )

    def create_session(
        self,
        *,
        session_id: str,
        turn_id: str,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
    ):
        return Qwen3LocalASRSession(
            engine=self,
            session_id=session_id,
            turn_id=turn_id,
            on_speech_started=on_speech_started,
            on_partial=on_partial,
            on_final=on_final,
            on_error=on_error,
        )


class Qwen3LocalASRSession:
    """Per-turn session for the local Qwen3-ASR engine.

    Streaming flow (vLLM backend):
      1. Browser PCM-16 chunks arrive; we VAD-gate them on energy.
      2. After speech starts we accumulate samples into a streaming-chunk
         buffer (~480 ms).  Each full buffer is fed to ``streaming_transcribe``
         off the asyncio thread, and any new prefix in ``state.text`` is
         emitted as an ASR partial.
      3. When silence persists for ``silence_limit_ms`` we drain the buffer,
         call ``finish_streaming_transcribe``, and emit the final transcript.

    Transformers backend behaves the same externally but only emits a single
    final transcript at end-of-utterance (no streaming partials).
    """

    def __init__(
        self,
        engine: Qwen3LocalASREngine,
        session_id: str,
        turn_id: str,
        on_speech_started: AsrSpeechStartedCallback,
        on_partial: AsrPartialCallback,
        on_final: AsrFinalCallback,
        on_error: AsrErrorCallback,
    ):
        self._engine = engine
        self._session_id = session_id
        self._turn_id = turn_id
        self._on_speech_started = on_speech_started
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error

        self._streaming_chunk_samples = int(SAMPLE_RATE * engine._streaming_chunk_ms / 1000)
        self._silence_limit_ms = engine._silence_limit_ms
        self._min_speech_samples = int(SAMPLE_RATE * engine._min_speech_ms / 1000)

        self._state = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._utterance_samples = 0
        self._silent_ms = 0.0
        self._is_speaking = False
        self._last_partial_text = ""
        self._last_partial_at_ms = 0
        self._first_audio_at_ms: int | None = None
        self._last_audio_at_ms: int | None = None
        self._speech_started_at_ms: int | None = None
        self._finalized = False
        self._cancelled = False

    async def start(self) -> None:
        if self._engine._backend == "vllm" and self._state is None:
            await asyncio.to_thread(self._init_streaming_state)
        # transformers / openai backends have no per-session warmup

    def _init_streaming_state(self) -> None:
        with self._engine._model_lock:
            self._state = self._engine._model.init_streaming_state(
                unfixed_chunk_num=self._engine._unfixed_chunk_num,
                unfixed_token_num=self._engine._unfixed_token_num,
                chunk_size_sec=self._engine._chunk_size_sec,
                language=self._engine._language,
            )

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self._cancelled or self._finalized or not pcm_bytes:
            return
        try:
            await self.start()
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return

            now_ms = time_ms()
            self._last_audio_at_ms = now_ms
            if self._first_audio_at_ms is None:
                self._first_audio_at_ms = now_ms
            chunk_ms = (samples.size / SAMPLE_RATE) * 1000.0
            rms = float(np.sqrt(np.mean(samples * samples))) * 32768.0

            if rms > self._engine._energy_threshold:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_started_at_ms = now_ms
                    await self._on_speech_started()
                self._silent_ms = 0.0
                self._append(samples)
            elif self._is_speaking:
                # Keep trailing silence in the buffer so the model sees natural
                # endings, but track silence to decide when to finalise.
                self._silent_ms += chunk_ms
                self._append(samples)
                if self._silent_ms >= self._silence_limit_ms:
                    await self._finalize_utterance(reason="vad-silence")
                    return

            # Drain whenever we have enough audio for a streaming step.
            if (
                self._engine._backend == "vllm"
                and self._is_speaking
                and self._buffer_samples >= self._streaming_chunk_samples
            ):
                await self._drain_streaming_chunk()
        except Exception as exc:  # pragma: no cover - runtime dependent
            log.exception(
                "Qwen3 local ASR send_audio failed session=%s turn=%s",
                self._session_id,
                self._turn_id,
            )
            await self._on_error(f"ASR failed: {exc}")

    async def finish(self) -> None:
        if self._cancelled or self._finalized:
            return
        try:
            await self._finalize_utterance(reason="client-finish")
        except Exception as exc:  # pragma: no cover - runtime dependent
            log.exception(
                "Qwen3 local ASR finish failed session=%s turn=%s",
                self._session_id,
                self._turn_id,
            )
            await self._on_error(f"ASR finish failed: {exc}")

    async def cancel(self) -> None:
        self._cancelled = True
        if self._state is not None and self._engine._backend == "vllm":
            try:
                await asyncio.to_thread(self._abort_state)
            except Exception:
                log.debug("Qwen3 local ASR cancel: state abort failed", exc_info=True)
        self._reset_buffers()

    # ── internals ────────────────────────────────────────────────────────

    def _append(self, samples: np.ndarray) -> None:
        self._buffer.append(samples)
        self._buffer_samples += samples.size
        self._utterance_samples += samples.size

    def _reset_buffers(self) -> None:
        self._buffer = []
        self._buffer_samples = 0

    def _abort_state(self) -> None:
        # qwen-asr has no public cancel; finishing on the worker is cheap.
        try:
            with self._engine._model_lock:
                self._engine._model.finish_streaming_transcribe(self._state)
        finally:
            self._state = None

    async def _drain_streaming_chunk(self) -> None:
        if not self._buffer:
            return
        chunk = np.concatenate(self._buffer)
        self._reset_buffers()
        await asyncio.to_thread(self._streaming_transcribe, chunk)
        text = _clean_qwen_asr_text(getattr(self._state, "text", ""))
        if not text or text == self._last_partial_text:
            return
        now_ms = time_ms()
        if (now_ms - self._last_partial_at_ms) < self._engine._partial_min_interval_ms:
            return
        self._last_partial_text = text
        self._last_partial_at_ms = now_ms
        await self._on_partial(text)

    def _streaming_transcribe(self, chunk: np.ndarray) -> None:
        with self._engine._model_lock:
            self._engine._model.streaming_transcribe(chunk, self._state)

    async def _finalize_utterance(self, *, reason: str) -> None:
        if self._finalized:
            return
        self._finalized = True

        # Drop ultra-short blips that are almost certainly noise.
        if self._utterance_samples < self._min_speech_samples and not self._last_partial_text:
            self._reset_buffers()
            return

        speech_ended_at_ms = self._last_audio_at_ms or time_ms()
        transcribe_started_at_ms = time_ms()

        text = ""
        if self._engine._backend == "vllm":
            # Push any tail audio still pending into the streaming state.
            if self._buffer:
                chunk = np.concatenate(self._buffer)
                self._reset_buffers()
                await asyncio.to_thread(self._streaming_transcribe, chunk)
            await asyncio.to_thread(self._finish_streaming)
            text = _clean_qwen_asr_text(getattr(self._state, "text", ""))
            self._state = None
        else:
            # transformers / openai backends: one-shot transcribe over the
            # whole utterance.
            audio = (
                np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
            )
            self._reset_buffers()
            if audio.size == 0:
                return
            if self._engine._backend == "openai":
                text = await asyncio.to_thread(self._openai_transcribe, audio)
            else:
                text = await asyncio.to_thread(self._transformers_transcribe, audio)

        transcribe_finished_at_ms = time_ms()

        if not text:
            log.debug(
                "Qwen3 local ASR finalised with empty transcript session=%s reason=%s",
                self._session_id,
                reason,
            )
            return

        timing = {
            "speechEndedAtMs": speech_ended_at_ms,
            "sttLatencyMs": transcribe_finished_at_ms - speech_ended_at_ms,
            "endpointWaitMs": transcribe_started_at_ms - speech_ended_at_ms,
            "transcribeDurationMs": transcribe_finished_at_ms - transcribe_started_at_ms,
        }
        await self._on_final(text, timing)

    def _finish_streaming(self) -> None:
        with self._engine._model_lock:
            self._engine._model.finish_streaming_transcribe(self._state)

    def _transformers_transcribe(self, audio: np.ndarray) -> str:
        with self._engine._model_lock:
            results = self._engine._model.transcribe(
                audio=(audio, SAMPLE_RATE),
                language=self._engine._language,
            )
        if not results:
            return ""
        text = getattr(results[0], "text", "") or ""
        return _clean_qwen_asr_text(text)

    def _openai_transcribe(self, audio: np.ndarray) -> str:
        """Send the buffered utterance to an external qwen-asr-serve instance.

        Uses the OpenAI-compatible chat-completions endpoint with an inline
        base64 wav data URL, then strips the standard ``<|...|>`` language
        prefix that qwen-asr emits.
        """
        engine = self._engine
        if not engine._openai_base_url:
            raise ASRUnavailableError(
                "openai backend selected but QWEN_LOCAL_OPENAI_BASE_URL is not configured",
            )
        try:
            import urllib.request
            import json as _json
        except Exception as exc:  # pragma: no cover - stdlib always present
            raise ASRUnavailableError(f"stdlib unavailable: {exc}") from exc

        wav_bytes = _wav_bytes_from_float32(audio, SAMPLE_RATE)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_url = f"data:audio/wav;base64,{audio_b64}"
        payload = {
            "model": engine._model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": engine._max_new_tokens,
        }
        req = urllib.request.Request(
            f"{engine._openai_base_url}/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {engine._openai_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=engine._openai_timeout_s) as resp:
                body = resp.read()
        except Exception as exc:
            raise ASRUnavailableError(f"qwen-asr-serve request failed: {exc}") from exc
        try:
            data = _json.loads(body.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ASRUnavailableError(f"qwen-asr-serve returned malformed response: {exc}") from exc
        return _clean_qwen_asr_text(content)
