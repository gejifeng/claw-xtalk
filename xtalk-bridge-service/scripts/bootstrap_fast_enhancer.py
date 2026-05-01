#!/usr/bin/env python3
"""
bootstrap_fast_enhancer.py
==========================

Download the k2-fsa sherpa-onnx GTCRN ("Fast Enhancer") speech-denoiser model
used by the bridge to filter ambient noise out of the microphone stream
*before* it reaches the ASR or the barge-in detector.

This is the single most effective fix for the failure mode where background
noise (HVAC, keyboard, distant TV, AI speaker bleed) keeps tripping the
barge-in path and interrupting the AI mid-sentence.

Usage:
    python scripts/bootstrap_fast_enhancer.py
    python scripts/bootstrap_fast_enhancer.py --out /custom/path/gtcrn_simple.onnx

The model is < 1 MB.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_OUT = ROOT_DIR / "pretrained_models" / "fast-enhancer" / "gtcrn_simple.onnx"

# Official release asset published by k2-fsa.
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"        →   {dest}")

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = done * 100 // total
        sys.stdout.write(f"\r  {pct:3d}%  {done // 1024} / {total // 1024} KB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        sys.exit(1)
    sys.stdout.write("\n")
    print(f"Done. {dest.stat().st_size // 1024} KB written.")


def _check_sherpa_onnx() -> None:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        print(
            "[WARN] sherpa-onnx is not installed in this environment.\n"
            "       Install it with:  pip install sherpa-onnx\n"
            "       (the model file will still be downloaded)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the GTCRN Fast Enhancer model used by the X-Talk bridge.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Destination .onnx file (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already exists.",
    )
    args = parser.parse_args()

    dest = Path(args.out)
    if dest.exists() and not args.force:
        print(f"[INFO] Already present: {dest}  ({dest.stat().st_size // 1024} KB)")
        print("       Pass --force to re-download.")
    else:
        _download(MODEL_URL, dest)

    _check_sherpa_onnx()

    print(
        "\nSetup complete.  Add the following to your .env (or rely on defaults):\n"
        "\n"
        "  SPEECH_ENHANCER_ENABLED=1\n"
        f"  SPEECH_ENHANCER_MODEL={dest}\n"
        "  SPEECH_ENHANCER_NUM_THREADS=1\n"
    )


if __name__ == "__main__":
    main()
