"""Transparent overlay window with Acrylic blur for Vietnamese subtitles.

Borderless, always-on-top window. Background is fully transparent —
only the subtitle text is visible, floating over a frosted-glass blur
of the Zoom meeting behind it.
Thread-safe, synchronous text updates.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_TP_COLOR = '#010101'
_OVERLAY_BOTTOM_MARGIN = int(os.environ.get('ZT_OVERLAY_BOTTOM_MARGIN', '80'))
_VI_FONT = 42
_JP_FONT = 20
_VI_FG = '#ffe066'
_JP_FG = '#cccccc'
_POLL_MS = 16
_FRAME_H = int(os.environ.get('ZT_OVERLAY_HEIGHT', '280'))


class OverlayDisplay:
    """Transparent subtitle overlay with Acrylic blur on Windows."""

    def __init__(self, *, x: int | None = None, y: int | None = None,
                 width: int | None = None, height: int = _FRAME_H) -> None:
        self._lock = threading.Lock()
        self._latest_jp = ""
        self._latest_vi = ""
        self._latest_info = ""
        self._root = None
        self._vi_label = None
        self._jp_label = None
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._ready = threading.Event()

        self._thread = threading.Thread(target=self._tkinter_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError(
                "Overlay window failed to initialise (no display server "
                "or tkinter unavailable). Run without --overlay."
            )

    def _tkinter_loop(self) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:
            logger.error("Overlay: tkinter not available: %s", exc)
            self._ready.set()
            return

        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        try:
            root = tk.Tk()
        except Exception as exc:
            logger.error("Overlay: cannot create window: %s", exc)
            self._ready.set()
            return

        root.title("")
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        root.configure(bg=_TP_COLOR)

        try:
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
        except Exception:
            sw, sh = 1920, 1080

        w = self._width or sw
        h = self._height
        px = self._x if self._x is not None else 0
        py = self._y if self._y is not None else sh - h - _OVERLAY_BOTTOM_MARGIN
        root.geometry(f"{w}x{h}+{px}+{py}")

        # Window-level transparency: _TP_COLOR pixels become fully transparent
        root.attributes('-transparentcolor', _TP_COLOR)

        # Apply Acrylic blur via Windows DWM
        _try_acrylic(root)

        # Japanese label — directly on root, fully transparent bg
        jp_label = tk.Label(
            root, text="",
            font=("Segoe UI", _JP_FONT),
            fg=_JP_FG, bg=_TP_COLOR,
            wraplength=w - 48,
            justify='center', anchor='s',
        )
        jp_label.pack(fill='x', expand=True, pady=(10, 0))

        # Vietnamese label — bold, bright, transparent bg
        vi_label = tk.Label(
            root, text="",
            font=("Segoe UI", _VI_FONT, "bold"),
            fg=_VI_FG, bg=_TP_COLOR,
            wraplength=w - 48,
            justify='center', anchor='n',
        )
        vi_label.pack(fill='x', expand=True, pady=(0, 10))

        self._vi_label = vi_label
        self._jp_label = jp_label
        self._root = root
        self._ready.set()
        self._poll_loop()
        root.mainloop()

    def _poll_loop(self) -> None:
        with self._lock:
            jp = self._latest_jp
            vi = self._latest_vi
            info = self._latest_info
            if info:
                self._latest_info = ""

        try:
            if info:
                if self._jp_label:
                    self._jp_label.config(text="")
                if self._vi_label:
                    self._vi_label.config(text=info)
            else:
                if self._jp_label and self._jp_label.cget("text") != jp:
                    self._jp_label.config(text=jp)
                if self._vi_label and self._vi_label.cget("text") != vi:
                    self._vi_label.config(text=vi)
        except Exception as exc:
            logger.error("overlay update: %s", exc)

        if self._root:
            self._root.after(_POLL_MS, self._poll_loop)

    # ─── Public API (thread-safe) ───────────────────────────────────────

    def show(self, japanese: str, vietnamese: str) -> None:
        with self._lock:
            self._latest_jp = japanese
            self._latest_vi = vietnamese
            self._latest_info = ""

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        with self._lock:
            self._latest_jp = japanese
            self._latest_vi = vietnamese
            self._latest_info = ""

    def show_source(self, japanese: str, seq: int | None = None) -> None:
        with self._lock:
            self._latest_jp = japanese
            self._latest_vi = ""
            self._latest_info = ""

    def show_target(self, vietnamese: str, japanese: str | None = None,
                    seq: int | None = None) -> None:
        with self._lock:
            if japanese:
                self._latest_jp = japanese
            self._latest_vi = vietnamese
            self._latest_info = ""

    def show_source_partial(self, committed: str, tail: str = "") -> None:
        text = committed
        if tail:
            text = f"{committed} [...]"
        with self._lock:
            self._latest_jp = text
            self._latest_vi = ""
            self._latest_info = ""

    def finalize_source(self, japanese: str) -> None:
        with self._lock:
            self._latest_jp = japanese
            self._latest_vi = ""
            self._latest_info = ""

    def info(self, message: str) -> None:
        with self._lock:
            self._latest_jp = ""
            self._latest_vi = ""
            self._latest_info = message

    def close(self) -> None:
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass


def _try_acrylic(root: Any) -> None:
    """Apply Acrylic blur behind the window on Windows 10+ (non-fatal)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        if not hwnd:
            return

        # Layered window + transparent (click-through) + no taskbar icon
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        WS_EX_TOOLWINDOW = 0x80
        current = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            current | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW,
        )

        class ACCENTPOLICY(ctypes.Structure):
            _fields_ = [
                ("AccentState", ctypes.c_uint),
                ("AccentFlags", ctypes.c_uint),
                ("GradientColor", ctypes.c_uint),
                ("AnimationId", ctypes.c_uint),
            ]

        class WINCOMPATTRDATA(ctypes.Structure):
            _fields_ = [
                ("Attribute", ctypes.c_int),
                ("Data", ctypes.POINTER(ACCENTPOLICY)),
                ("SizeOfData", ctypes.c_size_t),
            ]

        accent = ACCENTPOLICY()
        accent.AccentState = 4           # ACCENT_ENABLE_ACRYLIC
        accent.GradientColor = 0x50000000  # dark tint ~31% for text readability
        accent.AccentFlags = 2

        data = WINCOMPATTRDATA()
        data.Attribute = 19              # WCA_ACCENT_POLICY
        data.SizeOfData = ctypes.sizeof(accent)
        data.Data = ctypes.pointer(accent)

        ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.byref(data))
    except Exception as exc:
        logger.debug("Acrylic blur not available: %s", exc)
