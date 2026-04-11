import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(dotenv_path: Path) -> None:
	if not dotenv_path.is_file():
		return

	for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key or key in os.environ:
			continue

		if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
			value = value[1:-1]

		os.environ[key] = value


_load_dotenv(ROOT_DIR / ".env")

SIDECAR_HOST = os.getenv("SIDECAR_HOST", "127.0.0.1")
SIDECAR_PORT = int(os.getenv("SIDECAR_PORT", "7431"))

# ASR
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "qwen-realtime")
ASR_MODEL = os.getenv("WHISPER_MODEL_SIZE", "base")
ASR_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
ASR_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh")

# DashScope / Qwen Realtime ASR
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash-realtime")
QWEN_ASR_URL = os.getenv("QWEN_ASR_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
QWEN_ASR_LANGUAGE = os.getenv("QWEN_ASR_LANGUAGE", WHISPER_LANGUAGE)
QWEN_ASR_SAMPLE_RATE = int(os.getenv("QWEN_ASR_SAMPLE_RATE", "16000"))
QWEN_ASR_TURN_DETECTION_THRESHOLD = float(
	os.getenv("QWEN_ASR_TURN_DETECTION_THRESHOLD", "0.0"),
)
QWEN_ASR_TURN_DETECTION_SILENCE_MS = int(
	os.getenv("QWEN_ASR_TURN_DETECTION_SILENCE_MS", "400"),
)

# VAD
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "200.0"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "800"))
VAD_PARTIAL_INTERVAL_MS = int(os.getenv("VAD_PARTIAL_INTERVAL_MS", "1500"))

# TTS
TTS_ENABLED = os.getenv("TTS_ENABLED", "1").lower() not in {"0", "false", "no"}
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "aliyun-cosyvoice")

# Local CosyVoice
COSYVOICE_REPO_DIR = Path(os.getenv("COSYVOICE_REPO_DIR", str(ROOT_DIR / "vendor" / "CosyVoice")))
TTS_MODEL_DIR = Path(os.getenv("TTS_MODEL_DIR", str(ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B")))
TTS_MODE = os.getenv("TTS_MODE", "zero_shot")
TTS_SPEAKER_ID = os.getenv("TTS_SPEAKER_ID", "中文女")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_PROMPT_TEXT = os.getenv(
	"TTS_PROMPT_TEXT",
	"You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
)
TTS_PROMPT_WAV = Path(os.getenv("TTS_PROMPT_WAV", str(COSYVOICE_REPO_DIR / "asset" / "zero_shot_prompt.wav")))
TTS_INSTRUCT_TEXT = os.getenv("TTS_INSTRUCT_TEXT", "")

# DashScope / CosyVoice TTS
ALIYUN_COSYVOICE_MODEL = os.getenv("ALIYUN_COSYVOICE_MODEL", "cosyvoice-v3.5-flash")
ALIYUN_COSYVOICE_VOICE = os.getenv("ALIYUN_COSYVOICE_VOICE", "")
ALIYUN_COSYVOICE_AUDIO_FORMAT = os.getenv(
	"ALIYUN_COSYVOICE_AUDIO_FORMAT",
	"PCM_24000HZ_MONO_16BIT",
)
ALIYUN_COSYVOICE_VOLUME = int(os.getenv("ALIYUN_COSYVOICE_VOLUME", "50"))
ALIYUN_COSYVOICE_SPEECH_RATE = float(os.getenv("ALIYUN_COSYVOICE_SPEECH_RATE", str(TTS_SPEED)))
ALIYUN_COSYVOICE_PITCH_RATE = float(os.getenv("ALIYUN_COSYVOICE_PITCH_RATE", "1.0"))
ALIYUN_COSYVOICE_INSTRUCTION = os.getenv("ALIYUN_COSYVOICE_INSTRUCTION", "")
ALIYUN_COSYVOICE_TIMEOUT_MS = int(os.getenv("ALIYUN_COSYVOICE_TIMEOUT_MS", "30000"))
