#!/usr/bin/env python3
"""Entry point for the real-time Japanese -> Vietnamese Zoom translator.

Run from the project root:

    python3 main.py --list-devices
    python3 main.py --system-audio
    python3 main.py --device-index 3
"""
from __future__ import annotations

import argparse
import sys

import config
from src import audio_capture
from src.display import SubtitleDisplay


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


def main() -> int:
    args = parse_args()
    display = SubtitleDisplay()

    if args.list_devices:
        audio_capture.list_devices()
        return 0

    device = select_device(args, display)
    if device is None:
        return 1

    # Imported lazily so --list-devices works even before models are downloaded.
    from src.pipeline import TranslationPipeline

    try:
        pipeline = TranslationPipeline(device=device, display=display)
    except FileNotFoundError as exc:
        display.info(str(exc))
        return 1

    display.info(
        f"Listening... ({config.NLLB_SOURCE_LANG} -> {config.NLLB_TARGET_LANG}). "
        "Press Ctrl+C to stop."
    )
    pipeline.run_forever()
    display.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
