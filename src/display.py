"""Terminal display for bilingual Japanese/Vietnamese subtitles."""
from __future__ import annotations

import sys
import threading
import time

import config

# ANSI colors — disabled automatically when output is not a TTY.
_RESET = "\033[0m"
_DIM = "\033[2m"
_JP_COLOR = "\033[96m"   # bright cyan
_VI_COLOR = "\033[92m"   # bright green


def _color_enabled() -> bool:
    return bool(config.USE_COLOR) and sys.stdout.isatty()


class SubtitleDisplay:
    """Thread-safe terminal printer for translated subtitle pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._color = _color_enabled()

    def _wrap(self, text: str, color: str) -> str:
        if not self._color:
            return text
        return f"{color}{text}{_RESET}"

    def show(self, japanese: str, vietnamese: str) -> None:
        """Print one timestamped Japanese -> Vietnamese subtitle pair (one-shot)."""
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            print(f"\n{header}\n{jp_line}\n{vi_line}", flush=True)

    def show_source(self, japanese: str) -> None:
        """Print the recognized Japanese immediately, before translation is ready.

        Showing the source line as soon as ASR completes drastically cuts the
        *perceived* latency in a live meeting: the viewer sees what was just said
        within ~2 s, then the Vietnamese line follows when the translator finishes.
        """
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        with self._lock:
            print(f"\n{header}\n{jp_line}", flush=True)

    def show_target(self, vietnamese: str) -> None:
        """Print the Vietnamese line for the most recently shown source utterance."""
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            print(vi_line, flush=True)

    def info(self, message: str) -> None:
        """Print a status/diagnostic line."""
        with self._lock:
            print(self._wrap(message, _DIM) if self._color else message, flush=True)
