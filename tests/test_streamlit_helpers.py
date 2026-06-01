"""Script-style tests for the Streamlit dashboard's pure feed helpers.

These exercise webui/filters.py without importing Streamlit, so they run in the
plain test suite. Run: python tests/test_streamlit_helpers.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui.filters import filter_lines, latency_alert, prepare_feed, safe_stem  # noqa: E402


@dataclass
class _Line:
    seq: int
    jp: str
    vi: str


def _lines(n: int) -> list[_Line]:
    return [_Line(i, f"日本語{i}", f"Tiếng Việt {i}") for i in range(1, n + 1)]


def test_filter_empty_query_returns_all_as_new_list() -> None:
    src = _lines(3)
    out = filter_lines(src, "   ")
    assert out == src
    assert out is not src  # must be a copy, not the original


def test_filter_matches_jp_or_vi_case_insensitive() -> None:
    src = [_Line(1, "おはよう", "Chào buổi sáng"), _Line(2, "こんばんは", "Chào buổi tối")]
    assert [ln.seq for ln in filter_lines(src, "SÁNG")] == [1]
    assert [ln.seq for ln in filter_lines(src, "こんばんは")] == [2]
    assert filter_lines(src, "xyz") == []


def test_prepare_feed_tail_keeps_most_recent_chronological() -> None:
    out = prepare_feed(_lines(10), query="", tail=3, newest_first=False)
    assert [ln.seq for ln in out] == [8, 9, 10]


def test_prepare_feed_newest_first_reverses_same_slice() -> None:
    out = prepare_feed(_lines(10), query="", tail=3, newest_first=True)
    # Same lines as chronological tail, just reversed order.
    assert [ln.seq for ln in out] == [10, 9, 8]


def test_prepare_feed_tail_zero_means_no_limit() -> None:
    out = prepare_feed(_lines(5), query="", tail=0, newest_first=False)
    assert [ln.seq for ln in out] == [1, 2, 3, 4, 5]


def test_prepare_feed_filters_before_tail() -> None:
    src = [_Line(i, f"日本語{i}", "keep" if i % 2 == 0 else "drop") for i in range(1, 11)]
    out = prepare_feed(src, query="keep", tail=2, newest_first=False)
    # Even seqs match "keep": [2,4,6,8,10]; tail 2 -> [8,10].
    assert [ln.seq for ln in out] == [8, 10]


def test_latency_alert_threshold() -> None:
    assert latency_alert(1200.0, 1000.0) is True
    assert latency_alert(800.0, 1000.0) is False
    assert latency_alert(5000.0, 0) is False  # 0 disables the alert


def test_safe_stem_sanitizes_and_defaults() -> None:
    assert safe_stem("meeting 2026-06-01") == "meeting_2026-06-01"
    assert safe_stem("../../etc/passwd") == "etc_passwd"  # no path escape
    assert safe_stem("  ") == "transcript"  # empty -> default
    assert safe_stem("résumé!!") == "r_sum"  # unsafe chars collapsed, trimmed


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
