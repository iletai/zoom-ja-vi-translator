#!/usr/bin/env python3
"""Entry point for the real-time Japanese -> Vietnamese Zoom translator.

Run from the project root:

    python3 main.py --list-devices
    python3 main.py --system-audio
    python3 main.py --device-index 3
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# Windows console defaults to cp1252 which cannot display Japanese/Vietnamese.
# Force UTF-8 so print() works for CJK characters.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress noisy soundcard "data discontinuity" warnings — they fire when the
# audio capture thread misses a buffer (common during heavy model inference) but
# are not actionable and clutter the output.
# NOTE: soundcard.mediafoundation calls simplefilter('always', SoundcardRuntimeWarning)
# on import, so we must apply our filter AFTER importing soundcard.

import config
from src import audio_capture
from src.display import SubtitleDisplay

# Apply AFTER soundcard import to override its simplefilter('always')
warnings.filterwarnings("ignore", message="data discontinuity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Japanese->Vietnamese translator for Zoom meeting audio.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list available audio input/loopback devices and exit",
    )
    parser.add_argument(
        "--system-audio",
        action="store_true",
        help="auto-select the system loopback device (capture what you hear)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="explicit device index from --list-devices",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="use the online streaming recognizer for low-latency live captions "
        "(shows Japanese as it is spoken; slightly lower accuracy than the default)",
    )
    parser.add_argument(
        "--cloud",
        nargs="?",
        const="azure",
        default=None,
        choices=["azure"],
        metavar="PROVIDER",
        help="use a cloud speech-translation backend for the lowest latency "
        "(~0.5-1s). Currently 'azure' (Azure Speech Translation, JA->VI, 5 "
        "audio hours/month free). Requires AZURE_SPEECH_KEY and "
        "AZURE_SPEECH_REGION env vars and 'pip install -r requirements-cloud.txt'. "
        "Audio is sent to the provider; omit --cloud for fully offline local mode.",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="write a structured JSONL evidence log of every pipeline stage "
        "(asr/aggregator/enqueue/queue_drop/dedup_skip/translate/display) for "
        "debugging dropped data. Defaults to test_audio/evidence/run_<ts>.jsonl "
        "when given with no path. Also enabled via ZT_EVIDENCE_LOG=<path>.",
    )
    return parser.parse_args()


def select_device(args: argparse.Namespace, display: SubtitleDisplay):
    """Resolve which capture device to use based on CLI args."""
    devices = audio_capture.list_devices() if args.device_index is not None else None

    if args.device_index is not None:
        if devices is None or not (0 <= args.device_index < len(devices)):
            display.info(f"Invalid --device-index {args.device_index}.")
            return None
        device = devices[args.device_index]["device"]
        display.info(f"Using device [{args.device_index}]: {audio_capture._device_name(device)}")
        return device

    if args.system_audio:
        device = audio_capture.find_loopback_device()
        if device is None:
            display.info(
                "No loopback device found. On macOS install BlackHole "
                "(brew install blackhole-2ch); on Windows ensure audio plays "
                "through the default output. Falling back to the default microphone."
            )
            device = audio_capture.get_default_microphone()
        else:
            display.info(f"Using loopback device: {audio_capture._device_name(device)}")
        return device

    device = audio_capture.get_default_microphone()
    display.info(
        f"Using default microphone: {audio_capture._device_name(device)} "
        "(use --system-audio to capture Zoom output instead)."
    )
    return device


def _configure_evidence_log(args: argparse.Namespace, display: SubtitleDisplay):
    """Enable JSONL evidence logging from --log / ZT_EVIDENCE_LOG if requested.

    Returns the resolved log path (or ``None``) so the caller can auto-save a
    human-readable transcript next to it when the session ends.
    """
    from src import evidence_log

    path = args.log
    if path is None:
        path = config.EVIDENCE_LOG_PATH or None
    elif path == "":
        # --log with no argument: default to a timestamped file under evidence/.
        import time

        evidence_dir = config.PROJECT_ROOT / "test_audio" / "evidence"
        path = str(evidence_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    resolved = evidence_log.configure(path)
    if resolved is not None:
        display.info(f"Evidence log: {resolved}")
    return resolved


def _save_transcript(log_path, display: SubtitleDisplay) -> None:
    """Write a bilingual transcript (.txt + .srt) next to the evidence log.

    Best-effort: a failure to export must never mask the real session result, so
    every error is swallowed with a diagnostic line.
    """
    if not log_path:
        return
    try:
        import pathlib

        from src import transcript_export

        log = pathlib.Path(log_path)
        if not log.exists():
            return
        lines = transcript_export.build_lines(transcript_export.load_events(log))
        if not lines:
            return
        for fmt, ext in (("txt", ".txt"), ("srt", ".srt")):
            out = log.with_suffix(ext)
            out.write_text(transcript_export.render(lines, fmt), encoding="utf-8")
        display.info(
            f"Saved transcript: {log.with_suffix('.txt')} (+ .srt, {len(lines)} lines)"
        )
    except Exception as exc:  # pragma: no cover - best-effort save
        display.info(f"Transcript save skipped: {exc}")


def main() -> int:
    args = parse_args()
    display = SubtitleDisplay()

    if args.list_devices:
        audio_capture.list_devices()
        return 0

    _log_path = _configure_evidence_log(args, display)
    from src import evidence_log

    try:
        device = select_device(args, display)
        if device is None:
            return 1

        # Imported lazily so --list-devices works even before models are downloaded.
        from src.pipeline import TranslationPipeline

        backend = args.cloud if args.cloud else "local"
        try:
            pipeline = TranslationPipeline(
                device=device,
                display=display,
                streaming=args.streaming,
                backend=backend,
            )
        except FileNotFoundError as exc:
            display.info(str(exc))
            return 1
        except ImportError as exc:  # e.g. webrtcvad missing for VAD re-decode
            display.info(
                f"Missing dependency for offline re-decode: {exc}. "
                "Install webrtcvad or set ZT_NO_REDECODE=1 to use the online path."
            )
            return 1
        except ValueError as exc:  # e.g. missing cloud credentials
            display.info(str(exc))
            return 1

        if args.cloud:
            mode = f"cloud:{args.cloud}"
            langs = f"{config.CLOUD_SOURCE_LANG} -> {config.CLOUD_TARGET_LANG}"
        elif args.streaming:
            mode = "streaming"
            langs = f"{config.NLLB_SOURCE_LANG} -> {config.NLLB_TARGET_LANG}"
        else:
            mode = "offline"
            langs = f"{config.NLLB_SOURCE_LANG} -> {config.NLLB_TARGET_LANG}"
        display.info(
            f"Listening [{mode}]... ({langs}). Press Ctrl+C to stop."
        )
        pipeline.run_forever()
        display.info("Stopped.")
        return 0
    finally:
        # Always flush/close the evidence log and leave the terminal clean, even
        # when an error path (device selection, model load, audio failure) skips
        # the normal shutdown above. Closing first guarantees every event is on
        # disk before we read the log back to save the transcript.
        evidence_log.close()
        _save_transcript(_log_path, display)


if __name__ == "__main__":
    sys.exit(main())
