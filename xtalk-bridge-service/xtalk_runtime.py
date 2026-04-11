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