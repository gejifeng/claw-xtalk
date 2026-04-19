#!/usr/bin/env python3
"""
bootstrap_omnivoice.py – Download the OmniVoice model weights from HuggingFace.

Run once before starting the bridge service with TTS_PROVIDER=omnivoice.

Usage:
    python scripts/bootstrap_omnivoice.py [--repo k2-fsa/OmniVoice] [--out pretrained_models/OmniVoice]

For users in China who cannot reach huggingface.co directly:
    HF_ENDPOINT=https://hf-mirror.com python scripts/bootstrap_omnivoice.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Resolve project root (one level above scripts/) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_OUT = str(ROOT_DIR / "pretrained_models" / "OmniVoice")
DEFAULT_REPO = "k2-fsa/OmniVoice"


def _check_deps() -> None:
    missing = []
    for pkg in ("huggingface_hub", "omnivoice"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"[ERROR] Missing dependencies: {', '.join(missing)}\n"
            "  Install with: pip install omnivoice huggingface_hub"
        )
        sys.exit(1)


def _download(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"  Using HF mirror: {hf_endpoint}")

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} → {local_dir} …")
    t0 = time.perf_counter()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        # Exclude files that are only needed for the Gradio demo or large
        # example assets; the core model weights are always included.
        ignore_patterns=["*.md", "*.txt", "examples/*", "docs/*"],
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f} s  →  {local_dir}")


def _warmup(local_dir: Path) -> None:
    """Load the model once to trigger any one-time JIT/kernel compilation."""
    print("\nRunning warmup inference (this pre-compiles CUDA kernels) …")
    try:
        import torch
        from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    except ImportError as exc:
        print(f"[WARN] Cannot run warmup: {exc}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = OmniVoice.from_pretrained(
        str(local_dir),
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    cfg = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=2.0,
        denoise=True,
        position_temperature=5.0,
        class_temperature=0.0,
        preprocess_prompt=False,
        postprocess_output=False,
    )
    t0 = time.perf_counter()
    model.generate(text="热身推理。", generation_config=cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Warmup complete  ttfa={elapsed_ms:.0f} ms  device={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OmniVoice model weights.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Local destination directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the post-download warmup inference step",
    )
    args = parser.parse_args()

    _check_deps()

    local_dir = Path(args.out)
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[INFO] Model directory already populated: {local_dir}")
        print("  Delete it or choose a different --out path to re-download.")
    else:
        _download(args.repo, local_dir)

    if not args.skip_warmup:
        _warmup(local_dir)

    print(
        "\nSetup complete.  Add the following to your .env to activate OmniVoice TTS:\n"
        "\n"
        "  TTS_PROVIDER=omnivoice\n"
        f"  OMNIVOICE_MODEL_DIR={local_dir}\n"
        "  OMNIVOICE_NUM_STEP=8          # 8 = fastest (~250 ms TTFA), 16 = higher quality\n"
        "  # Optional voice cloning:\n"
        "  # OMNIVOICE_REF_AUDIO=reference-audio/my_voice.wav\n"
        "  # OMNIVOICE_REF_TEXT=reference audio transcript here\n"
        "  # Optional voice design (if no ref audio):\n"
        "  # OMNIVOICE_INSTRUCT=female, low pitch\n"
    )


if __name__ == "__main__":
    main()
