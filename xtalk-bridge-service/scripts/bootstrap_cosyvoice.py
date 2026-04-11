#!/usr/bin/env python3
"""Clone CosyVoice and download Fun-CosyVoice3 model assets."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TTSFRD_ID = "FunAudioLLM/CosyVoice-ttsfrd"


def run_command(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        run_command(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"])
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", "--recursive", REPO_URL, str(repo_dir)])
    run_command(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"])


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare CosyVoice runtime for xtalk-bridge-service")
    parser.add_argument("--repo-dir", default=str(root_dir / "vendor" / "CosyVoice"))
    parser.add_argument("--model-dir", default=str(root_dir / "pretrained_models" / "Fun-CosyVoice3-0.5B"))
    parser.add_argument("--download-ttsfrd", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    model_dir = Path(args.model_dir)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    ensure_repo(repo_dir)

    if args.install_deps:
        run_command([sys.executable, "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")])

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=str(model_dir))

    if args.download_ttsfrd:
        snapshot_download(TTSFRD_ID, local_dir=str(root_dir / "pretrained_models" / "CosyVoice-ttsfrd"))

    print("CosyVoice bootstrap complete.")
    print(f"COSYVOICE_REPO_DIR={repo_dir}")
    print(f"TTS_MODEL_DIR={model_dir}")
    print("If this is a fresh environment, start the sidecar after installing CosyVoice dependencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())