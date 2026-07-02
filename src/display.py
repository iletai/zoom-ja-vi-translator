"""Terminal display for bilingual Japanese/Vietnamese subtitles."""
from __future__ import annotations

import ctypes
import logging
import shutil
import sys
import threading
import time
import unicodedata

import config

logger = logging.getLogger(__name__)

# ANSI colors — disabled automatically when output is not a TTY.
_RESET = "\033[0m"
_DIM = "\033[2m"
_JP_COLOR = "\033[96m"   # bright cyan
_VI_COLOR = "\033[92m"   # bright green


def _color_enabled() -> bool:
    return bool(config.USE_COLOR) and sys.stdout.isatty()


# ─── Windows Console auto-scroll helper ──────────────────────────────────
# When the user scrolls up in a Windows terminal, new output is written at
# the bottom of the buffer but the viewport stays put. This helper forces
# the viewport to follow the cursor so the latest subtitle is always visible.

if sys.platform == "win32":
    import ctypes.wintypes

    _STD_OUTPUT_HANDLE = -11

    class _COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.wintypes.SHORT), ("Y", ctypes.wintypes.SHORT)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [
            ("Left", ctypes.wintypes.SHORT),
            ("Top", ctypes.wintypes.SHORT),
            ("Right", ctypes.wintypes.SHORT),
            ("Bottom", ctypes.wintypes.SHORT),
        ]

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [
            ("dwSize", _COORD),
            ("dwCursorPosition", _COORD),
            ("wAttributes", ctypes.wintypes.WORD),
            ("srWindow", _SMALL_RECT),
            ("dwMaximumWindowSize", _COORD),
        ]


def _scroll_to_bottom() -> None:
    """Scroll the console viewport so the cursor (latest output) is visible."""
    if sys.platform != "win32" or not sys.stdout.isatty():
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(csbi)):
            return
        window_height = csbi.srWindow.Bottom - csbi.srWindow.Top
        cursor_y = csbi.dwCursorPosition.Y
        # Only scroll if the cursor is below the current viewport
        if cursor_y > csbi.srWindow.Bottom:
            new_top = cursor_y - window_height
            rect = _SMALL_RECT(
                csbi.srWindow.Left,
                ctypes.wintypes.SHORT(new_top),
                csbi.srWindow.Right,
                ctypes.wintypes.SHORT(cursor_y),
            )
            kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(rect))
    except Exception:
        pass


class SubtitleDisplay:
    """Thread-safe terminal printer for translated subtitle pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._color = _color_enabled()
        self._isatty = sys.stdout.isatty()
        self._auto_scroll = bool(config.AUTO_SCROLL) and self._isatty
        # True while an in-progress streaming partial is sitting on the current
        # terminal line (printed without a newline). Any method that prints a
        # committed line must clear it first so the two don't run together.
        self._partial_active = False
        # seq of the most recent show_source() line that has not yet had its
        # paired target printed. Lets show_target keep the JP/VI pair together
        # even when batching prints several sources before the first translation.
        self._last_source_seq: int | None = None
        # Terminal width cached at startup — get_terminal_size() is called on
        # every ASR-token tick in streaming mode, syscall overhead adds up.
        self._cols: int = shutil.get_terminal_size((80, 24)).columns
        # Partial-line budget is fixed (prefix + cols are constant); compute once
        # instead of re-measuring the prefix on every streaming ASR tick.
        self._partial_avail: int = max(0, self._cols - self._display_width("  JP… ") - 1)

    def _maybe_scroll(self) -> None:
        """Scroll terminal viewport to latest output if auto-scroll is enabled."""
        if self._auto_scroll:
            _scroll_to_bottom()

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
        logger.info("JP: %s | VI: %s", japanese, vietnamese)
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            print(f"{self._clear_partial()}\n{header}\n{jp_line}\n{vi_line}", flush=True)
            self._last_source_seq = None
            self._maybe_scroll()

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        """Atomically print one committed Japanese -> Vietnamese subtitle pair."""
        logger.info("JP: %s | VI: %s", japanese, vietnamese)
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            clear = self._clear_partial()
            print(f"{clear}\n{header}\n{jp_line}\n{vi_line}", flush=True)
            self._last_source_seq = None
            self._maybe_scroll()

    def show_source(self, japanese: str, seq: int | None = None) -> None:
        """Print the recognized Japanese immediately, before translation is ready."""
        logger.info("ASR: %s (seq=%s)", japanese, seq)
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        with self._lock:
            print(f"{self._clear_partial()}\n{header}\n{jp_line}", flush=True)
            self._last_source_seq = seq
            self._maybe_scroll()

    def show_target(
        self, vietnamese: str, japanese: str | None = None, seq: int | None = None
    ) -> None:
        """Print the Vietnamese line for a previously shown source utterance."""
        logger.info("VI: %s (seq=%s)", vietnamese, seq)
        vi_line = "  " + self._wrap(f"VI {vietnamese}", _VI_COLOR)
        with self._lock:
            superseded = (
                seq is not None and seq != self._last_source_seq and japanese is not None
            )
            if superseded:
                timestamp = time.strftime("%H:%M:%S")
                header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
                jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
                print(f"{self._clear_partial()}\n{header}\n{jp_line}\n{vi_line}", flush=True)
            else:
                print(f"{self._clear_partial()}{vi_line}", flush=True)
            self._last_source_seq = None
            self._maybe_scroll()

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
        avail = self._partial_avail  # spare column avoids wrap; precomputed at init
        jp_line = self._partial_line(committed, tail, avail)
        with self._lock:
            # \r returns to column 0; \033[K clears to end of line.
            print(f"\r\033[K  {jp_line}", end="", flush=True)
            self._partial_active = True
            self._maybe_scroll()

    def finalize_source(self, japanese: str) -> None:
        """Commit the streamed Japanese line (print the final text + newline)."""
        timestamp = time.strftime("%H:%M:%S")
        header = self._wrap(f"[{timestamp}]", _DIM) if self._color else f"[{timestamp}]"
        jp_line = "  " + self._wrap(f"JP {japanese}", _JP_COLOR)
        with self._lock:
            # Clear the in-progress partial line, then print the committed pair.
            print(f"{self._clear_partial()}{header}\n{jp_line}", flush=True)
            self._last_source_seq = None
            self._maybe_scroll()

    def info(self, message: str) -> None:
        """Print a status/diagnostic line."""
        logger.info(message)
        with self._lock:
            text = self._wrap(message, _DIM) if self._color else message
            print(f"{self._clear_partial()}{text}", flush=True)
            self._last_source_seq = None
            self._maybe_scroll()
