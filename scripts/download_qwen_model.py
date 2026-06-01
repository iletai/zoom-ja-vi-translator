#!/usr/bin/env python3
"""Download Qwen2.5-3B-Instruct GGUF model for LLM translation backend."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import certifi
except ImportError:
    certifi = None
else:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
MODEL_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
REPO_ID = "bartowski/Qwen2.5-3B-Instruct-GGUF"


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-3B model for LLM translation")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        help="Quantization variant (default: Q4_K_M)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_file = MODEL_FILE if args.quant == "Q4_K_M" else f"Qwen2.5-3B-Instruct-{args.quant}.gguf"
    model_path = MODEL_DIR / model_file

    if model_path.exists() and not args.force:
        print(f"✓ Model already exists: {model_path} ({format_size(model_path.stat().st_size)})")
        print("  Use --force to re-download.")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    print(f"Downloading {REPO_ID} / {model_file}...")
    print(f"Destination: {model_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "✗ Missing dependency 'huggingface_hub'. Install dependencies with: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=model_file,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            force_download=args.force,
        )
        downloaded_path = Path(downloaded)
        print(f"\n✓ Download complete: {downloaded_path} ({format_size(downloaded_path.stat().st_size)})")
        return 0
    except KeyboardInterrupt:
        print("\nDownload interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  1. Check internet connection", file=sys.stderr)
        print("  2. Install certifi: pip install certifi", file=sys.stderr)
        print(f"  3. Try manual download from https://huggingface.co/{REPO_ID}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
