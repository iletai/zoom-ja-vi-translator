#!/usr/bin/env python3
"""Download and prepare ASR and translation models for the Zoom translator."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This script downloads models from the HF Hub, so it must always run online —
# even when models already exist (e.g. a --force re-download). Opt out of the
# offline mode config enables for the runtime app before importing config.
os.environ["ZT_HF_ONLINE"] = "1"

import config  # noqa: E402

try:
    import huggingface_hub  # noqa: E402
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    huggingface_hub = None

# ReazonSpeech k2-v2 sherpa-onnx model verified on Hugging Face. It contains
# encoder/decoder/joiner ONNX Transducer files plus tokens.txt. If this repo is
# ever moved, change only this constant.
ASR_HF_REPO = "reazon-research/reazonspeech-k2-v2"

# Multilingual streaming (online) zipformer used by the optional --streaming mode.
# Emits partial hypotheses for low-latency live captions; covers Japanese.
STREAMING_ASR_HF_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"
STREAMING_ASR_ALLOW_PATTERNS = [
    "*chunk-16-left-128*.onnx",
    "tokens.txt",
]

# Canonical sherpa-onnx int8 configuration for this model: the encoder and
# joiner are int8-quantized, but the decoder stays fp32 (.onnx, not .int8.onnx).
# Downloading these four files (~150 MB) avoids pulling the full ~775 MB repo.
ASR_ALLOW_PATTERNS = [
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
    "README.md",
]
TOKENIZER_ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
]


class ModelDownloadError(RuntimeError):
    """Raised when a model download or conversion cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/prepare ReazonSpeech ASR and NLLB CTranslate2 models."
    )
    parser.add_argument("--asr-only", action="store_true", help="download only the ASR model")
    parser.add_argument("--nllb-only", action="store_true", help="download/prepare only the NLLB model")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="also download the multilingual streaming ASR model (for --streaming mode)",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="convert facebook/nllb-200-distilled-600M locally instead of downloading the pre-converted CT2 repo",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing target directories and download/convert again",
    )
    args = parser.parse_args()

    if args.asr_only and args.nllb_only:
        parser.error("--asr-only and --nllb-only cannot be used together")

    return args


def require_huggingface_hub() -> None:
    if huggingface_hub is None:
        raise ModelDownloadError(
            "Missing dependency 'huggingface_hub'. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        )


def is_non_empty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def prepare_target_dir(path: Path, force: bool) -> bool:
    """Return True if the caller should populate path, False if it can be skipped."""
    if is_non_empty_dir(path):
        if not force:
            print(f"Skipping {path}: already exists and is non-empty (use --force to re-download).")
            return False
        print(f"Removing existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return True


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def list_relative_files(path: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(pattern))
    return sorted({file.relative_to(path) for file in files})


def print_asr_artifacts(path: Path) -> None:
    artifacts = list_relative_files(path, ("*.onnx", "tokens.txt"))
    if not artifacts:
        print("Warning: no ASR .onnx or tokens.txt files found after download.")
        return

    print("ASR artifacts:")
    for artifact in artifacts:
        print(f"  - {artifact}")


def snapshot_download(repo_id: str, local_dir: Path | None = None, **kwargs: object) -> str:
    require_huggingface_hub()
    try:
        return huggingface_hub.snapshot_download(  # type: ignore[union-attr]
            repo_id=repo_id,
            local_dir=str(local_dir) if local_dir is not None else None,
            local_dir_use_symlinks=False,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - exact HF exception types vary by version
        target = f" into {local_dir}" if local_dir is not None else ""
        raise ModelDownloadError(
            f"Failed to download Hugging Face repo '{repo_id}'{target}: {exc}\n"
            "Check your network connection, Hugging Face access, and that the repo id is correct."
        ) from exc


def download_asr(force: bool) -> str:
    print(f"\n[ASR] ReazonSpeech k2-v2 sherpa-onnx repo: {ASR_HF_REPO}")
    if not prepare_target_dir(config.ASR_MODEL_DIR, force):
        print_asr_artifacts(config.ASR_MODEL_DIR)
        return "skipped"

    snapshot_download(
        repo_id=ASR_HF_REPO,
        local_dir=config.ASR_MODEL_DIR,
        allow_patterns=ASR_ALLOW_PATTERNS,
    )
    print_asr_artifacts(config.ASR_MODEL_DIR)
    print(f"[ASR] Done: {config.ASR_MODEL_DIR} ({format_size(directory_size(config.ASR_MODEL_DIR))})")
    return "downloaded"


def pre_cache_nllb_tokenizer() -> None:
    print(f"[NLLB] Pre-caching tokenizer files from {config.NLLB_HF_MODEL}")
    snapshot_download(repo_id=config.NLLB_HF_MODEL, allow_patterns=TOKENIZER_ALLOW_PATTERNS)


def download_nllb_preconverted(force: bool) -> str:
    print(f"\n[NLLB] Pre-converted CTranslate2 repo: {config.NLLB_CT2_HF_REPO}")
    if not prepare_target_dir(config.NLLB_CT2_DIR, force):
        return "skipped"

    snapshot_download(repo_id=config.NLLB_CT2_HF_REPO, local_dir=config.NLLB_CT2_DIR)
    pre_cache_nllb_tokenizer()
    print(f"[NLLB] Done: {config.NLLB_CT2_DIR} ({format_size(directory_size(config.NLLB_CT2_DIR))})")
    return "downloaded"


def convert_nllb(force: bool) -> str:
    print(f"\n[NLLB] Local int8 conversion from {config.NLLB_HF_MODEL}")
    if not prepare_target_dir(config.NLLB_CT2_DIR, force):
        return "skipped"

    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise ModelDownloadError(
            "Missing 'ct2-transformers-converter'. Install CTranslate2 with: "
            "python -m pip install ctranslate2"
        )

    command = [
        converter,
        "--model",
        config.NLLB_HF_MODEL,
        "--output_dir",
        str(config.NLLB_CT2_DIR),
        "--quantization",
        "int8",
        "--copy_files",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
    ]
    print("[NLLB] Running:")
    print("  " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ModelDownloadError("ct2-transformers-converter was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise ModelDownloadError(
            f"NLLB conversion failed with exit code {exc.returncode}. "
            "Check available disk space, network access, and CTranslate2/Transformers installation."
        ) from exc

    pre_cache_nllb_tokenizer()
    print(f"[NLLB] Done: {config.NLLB_CT2_DIR} ({format_size(directory_size(config.NLLB_CT2_DIR))})")
    return "converted"


def download_streaming_asr(force: bool) -> str:
    print(f"\n[Streaming ASR] sherpa-onnx online zipformer repo: {STREAMING_ASR_HF_REPO}")
    if not prepare_target_dir(config.STREAMING_ASR_MODEL_DIR, force):
        print_asr_artifacts(config.STREAMING_ASR_MODEL_DIR)
        return "skipped"

    snapshot_download(
        repo_id=STREAMING_ASR_HF_REPO,
        local_dir=config.STREAMING_ASR_MODEL_DIR,
        allow_patterns=STREAMING_ASR_ALLOW_PATTERNS,
    )
    print_asr_artifacts(config.STREAMING_ASR_MODEL_DIR)
    print(
        f"[Streaming ASR] Done: {config.STREAMING_ASR_MODEL_DIR} "
        f"({format_size(directory_size(config.STREAMING_ASR_MODEL_DIR))})"
    )
    return "downloaded"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    statuses: dict[str, str] = {}
    try:
        if not args.nllb_only:
            statuses["ASR"] = download_asr(force=args.force)
        if not args.asr_only:
            statuses["NLLB"] = convert_nllb(force=args.force) if args.convert else download_nllb_preconverted(force=args.force)
        if args.streaming:
            statuses["Streaming ASR"] = download_streaming_asr(force=args.force)
    except ModelDownloadError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    print("\nSummary:")
    for name, status in statuses.items():
        print(f"  - {name}: {status}")
    print(f"  - Models directory: {config.MODELS_DIR}")
    print(f"  - Total disk size: {format_size(directory_size(config.MODELS_DIR))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
