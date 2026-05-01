#!/usr/bin/env python3
"""
Download Qwen3-ASR-0.6B (or any other Qwen3-ASR checkpoint) into a local cache
so the sidecar can start fully offline.

Usage:
    python scripts/bootstrap_qwen3_asr.py
    python scripts/bootstrap_qwen3_asr.py --model Qwen/Qwen3-ASR-1.7B
    python scripts/bootstrap_qwen3_asr.py --target ./pretrained_models/Qwen3-ASR-0.6B

For users in Mainland China, set HF_ENDPOINT=https://hf-mirror.com before
running, or pass --use-modelscope to download from ModelScope instead.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


def _hf_download(model: str, target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface-hub is not installed; run: pip install -U huggingface_hub")
    target.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap_qwen3_asr] Downloading {model} -> {target} via Hugging Face")
    snapshot_download(repo_id=model, local_dir=str(target))


def _modelscope_download(model: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap_qwen3_asr] Downloading {model} -> {target} via ModelScope")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "modelscope",
            "download",
            "--model",
            model,
            "--local_dir",
            str(target),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repo id (default: %(default)s)")
    parser.add_argument(
        "--target",
        default=None,
        help="Local directory to download into (default: ./pretrained_models/<model-basename>)",
    )
    parser.add_argument("--use-modelscope", action="store_true", help="Use ModelScope instead of Hugging Face")
    args = parser.parse_args()

    target = (
        Path(args.target).resolve()
        if args.target
        else ROOT_DIR / "pretrained_models" / Path(args.model).name
    )

    if args.use_modelscope:
        _modelscope_download(args.model, target)
    else:
        _hf_download(args.model, target)

    print(
        "[bootstrap_qwen3_asr] Done. Point QWEN_LOCAL_MODEL at the local "
        f"directory if you want to skip auto-download:\n  QWEN_LOCAL_MODEL={target}"
    )


if __name__ == "__main__":
    main()
