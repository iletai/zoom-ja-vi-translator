"""Regression tests for the streaming subtitle display.

These guard the two live-caption rendering bugs seen in real Zoom runs:

1. A long Japanese partial wrapped past the terminal width, so the ``\\r``-based
   in-place overwrite could only clear the last wrapped row and stale fragments
   piled up into garbage like ``JP… x  JP… x  JP… x``.
2. The asynchronous Vietnamese line (``show_target``) was printed without first
   clearing the in-progress partial, so it ran onto the same line:
   ``JP… 今  VI Tôi muốn...``.

Run: ``python3 tests/test_display_streaming.py``
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.display import SubtitleDisplay  # noqa: E402

CLEAR = "\r\033[K"
LONG_JP = "なるほどでもせっかくなんですがこの間の素材は評判がよかったので色だけ変えてほしい"


def _display(isatty: bool) -> SubtitleDisplay:
    d = SubtitleDisplay()
    d._color = False          # disable ANSI colour for simple string assertions
    d._isatty = isatty
    return d


def test_truncation_never_exceeds_terminal_width() -> None:
    for max_width in range(1, 40):
        out = SubtitleDisplay._truncate_tail(LONG_JP, max_width)
        assert SubtitleDisplay._display_width(out) <= max_width, (max_width, out)


def test_truncation_keeps_newest_tail() -> None:
    # When truncated, the marker is prepended and the *end* of the text is kept.
    out = SubtitleDisplay._truncate_tail(LONG_JP, 12)
    assert out.startswith("…")
    assert LONG_JP.endswith(out[1:])


def test_partial_stays_on_one_physical_line() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_source_partial(LONG_JP)
    text = buf.getvalue()
    # No newline (in-place line) and the visible portion never wraps.
    assert "\n" not in text
    visible = text.replace(CLEAR, "").lstrip()
    assert SubtitleDisplay._display_width(visible) < 200  # fits a normal terminal


def test_target_clears_pending_partial() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_source_partial("今")
        d.show_target("Tôi muốn biết")
    text = buf.getvalue()
    # The VI line must be preceded by a clear sequence so it does not glue onto
    # the partial (the "JP… 今  VI ..." bug).
    vi_index = text.index("VI Tôi muốn biết")
    assert CLEAR in text[:vi_index]
    assert "今  VI" not in text


def test_finalize_clears_pending_partial() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_source_partial("今")
        d.finalize_source("今日はいい天気です")
    text = buf.getvalue()
    assert text.startswith(CLEAR)
    assert "JP 今日はいい天気です" in text


def test_non_tty_skips_partials() -> None:
    d = _display(isatty=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_source_partial(LONG_JP)
    # No control characters / partial output when not attached to a terminal.
    assert buf.getvalue() == ""


def test_show_pair_prints_jp_then_vi_contiguously() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_pair("がよかったので", "Và điều đó rất tốt")
    text = buf.getvalue()
    jp_index = text.index("  JP がよかったので")
    vi_index = text.index("  VI Và điều đó rất tốt")
    assert jp_index < vi_index
    assert "  JP がよかったので\n  VI Và điều đó rất tốt" in text


def test_show_pair_clears_pending_partial() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_source_partial("が")
        d.show_pair("が", "Nhưng")
    text = buf.getvalue()
    jp_index = text.index("JP が")
    assert CLEAR in text[:jp_index]
    assert d._partial_active is False


def test_show_pair_without_color_has_no_ansi_codes() -> None:
    d = _display(isatty=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        d.show_pair("こんにちは", "Xin chào")
    text = buf.getvalue()
    assert "\033[" not in text
    assert "JP こんにちは" in text
    assert "VI Xin chào" in text


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
