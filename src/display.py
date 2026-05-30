"""Terminal display for bilingual Japanese/Vietnamese subtitles."""
from __future__ import annotations

import shutil
import sys
import threading
import time
import unicodedata

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
        self._isatty = sys.stdout.isatty()
        # True while an in-progress streaming partial is sitting on the current
        # terminal line (printed without a newline). Any method that prints a
        # committed line must clear it first so the two don't run together.
        self._partial_active = False

    @staticmethod
    def _char_width(ch: str) -> int:
        """Display columns a character occupies (CJK/full-width count as 2)."""
        return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

    @classmethod
    def _display_width(cls, text: str) -> int:
        return sum(cls._char_width(ch) for ch in text)

    @classmethod
    def _truncate_tail(cls, text: str, max_width: int) -> str:
        """Keep the rightmost characters of ``text`` that fit in ``max_width`` columns.

        The newest words in a live partial are at the end, so showing the tail
        keeps the most relevant text visible while guaranteeing the line never
        wraps (which would break the ``\\r``-based in-place overwrite).
        """
        if max_width <= 0:
            return ""
        if cls._display_width(text) <= max_width:
            return text
        # Truncated: reserve the marker's own width so the returned string
        # (marker + tail) still fits within max_width regardless of how the
        # terminal counts the ellipsis.
        marker = "…"
        budget = max_width - cls._char_width(marker)
        if budget <= 0:
            return marker
        width = 0
        out: list[str] = []
        for ch in reversed(text):
            cw = cls._char_width(ch)
            if width + cw > budget:
                break
            out.append(ch)
            width += cw
        return marker + "".join(reversed(out))

    def _wrap(self, text: str, color: str) -> str:
        if not self._color:
            return text
        return f"{color}{text}{_RESET}"

    def _clear_partial(self) -> str:
        """Return a prefix that erases a pending in-place partial line.

        Must be called while holding ``self._lock``. Resets the pending flag so
        the next committed line starts clean.
        """
        if self._isatty and self._partial_active:
            self._partial_active = False
            return "\r\033[K"
        return ""

    def show(self, japanese: str, vietnamese: str) -> None:
        """Print one timestamped Japanese -> Vietnamese subtitle pair (one-shot)."""
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            print(f"{self._clear_partial()}\n{header}\n{jp_line}\n{vi_line}", flush=True)

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        """Atomically print one committed Japanese -> Vietnamese subtitle pair."""
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            clear = self._clear_partial()
            print(f"{clear}\n{header}\n{jp_line}\n{vi_line}", flush=True)

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
            print(f"{self._clear_partial()}\n{header}\n{jp_line}", flush=True)

    def show_target(self, vietnamese: str) -> None:
        """Print the Vietnamese line for the most recently shown source utterance."""
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            print(f"{self._clear_partial()}{vi_line}", flush=True)

    @classmethod
    def _truncate_segments(cls, committed: str, tail: str, max_width: int) -> tuple[str, str, str]:
        """Truncate committed/tail text as one line while preserving segments."""
        text = committed + tail
        shown = cls._truncate_tail(text, max_width)
        marker = "…" if shown.startswith("…") and not text.startswith(shown) else ""
        visible = shown[1:] if marker else shown
        committed_start = max(0, len(text) - len(visible))
        committed_end = len(committed)
        shown_committed = visible[: max(0, committed_end - committed_start)]
        shown_tail = visible[len(shown_committed) :]
        return marker, shown_committed, shown_tail

    def _partial_line(self, committed: str, tail: str, max_width: int) -> str:
        marker, shown_committed, shown_tail = self._truncate_segments(committed, tail, max_width)
        if not self._color:
            return f"JP… {marker}{shown_committed}{shown_tail}"
        solid = f"JP… {marker}{shown_committed}"
        if not shown_tail:
            return self._wrap(solid, _JP_COLOR)
        return f"{self._wrap(solid, _JP_COLOR)}{_DIM}{shown_tail}{_RESET}"

    def show_source_partial(self, committed: str, tail: str = "") -> None:
        """Overwrite the current line with stable + volatile in-progress Japanese.

        ``committed`` is shown in the normal Japanese colour and will not be
        rewritten by the streaming pipeline. ``tail`` is still volatile ASR text
        and is rendered dim. Both segments are truncated together to the terminal
        width so the partial always occupies a single physical line.
        """
        if not committed and not tail:
            return
        # When output is not a terminal (piped/redirected), in-place rewriting
        # produces control-character junk; skip partials and rely on finals.
        if not self._isatty:
            return
        prefix = "  JP… "
        cols = shutil.get_terminal_size((80, 24)).columns
        avail = max(0, cols - self._display_width(prefix) - 1)  # spare column avoids wrap
        jp_line = self._partial_line(committed, tail, avail)
        with self._lock:
            # \r returns to column 0; \033[K clears to end of line.
            print(f"\r\033[K  {jp_line}", end="", flush=True)
            self._partial_active = True

    def finalize_source(self, japanese: str) -> None:
        """Commit the streamed Japanese line (print the final text + newline)."""
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        with self._lock:
            # Clear the in-progress partial line, then print the committed pair.
            print(f"{self._clear_partial()}{header}\n{jp_line}", flush=True)

    def info(self, message: str) -> None:
        """Print a status/diagnostic line."""
        with self._lock:
            text = self._wrap(message, _DIM) if self._color else message
            print(f"{self._clear_partial()}{text}", flush=True)
