#!/usr/bin/env python3
"""Download Qwen2.5 GGUF model for LLM translation backend."""
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

# Model variants: 1.5B is recommended for ≤16GB RAM systems
MODELS = {
    "1.5b": {
        "repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "dir": PROJECT_ROOT / "models" / "qwen2.5-1.5b-instruct",
        "file": "Qwen2.5-1.5B-Instruct-{quant}.gguf",
        "size_hint": "~0.9 GB",
    },
    "3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "dir": PROJECT_ROOT / "models" / "qwen2.5-3b-instruct",
        "file": "Qwen2.5-3B-Instruct-{quant}.gguf",
        "size_hint": "~1.9 GB",
    },
}


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2.5 model for LLM translation")
    parser.add_argument(
        "--size", choices=["1.5b", "3b"], default="1.5b",
        help="Model size: 1.5b (recommended for ≤16GB RAM) or 3b (default: 1.5b)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument(
        "--quant", default="Q4_K_M",
        help="Quantization variant (default: Q4_K_M)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_info = MODELS[args.size]
    model_file = model_info["file"].format(quant=args.quant)
    model_dir = model_info["dir"]
    model_path = model_dir / model_file

    if model_path.exists() and not args.force:
        print(f"✓ Model already exists: {model_path} ({format_size(model_path.stat().st_size)})")
        print("  Use --force to re-download.")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    print(f"Downloading Qwen2.5-{args.size.upper()} ({model_info['size_hint']})...")
    print(f"  Repo: {model_info['repo']}")
    print(f"  File: {model_file}")
    print(f"  Dest: {model_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "✗ Missing dependency 'huggingface_hub'. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        downloaded = hf_hub_download(
            repo_id=model_info["repo"],
            filename=model_file,
            local_dir=str(model_dir),
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
        print(f"  3. Try manual download from https://huggingface.co/{model_info['repo']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
