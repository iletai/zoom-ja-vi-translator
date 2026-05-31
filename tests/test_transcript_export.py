"""Tests for src/transcript_export.py — transcript reconstruction & rendering."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import transcript_export as te


def _sample_events() -> list[dict]:
    # Two utterances, 1:1 segment_finalized/translate/display in seq order.
    return [
        {"event": "session_start", "t_ms": 0.0, "ts": "10:00:00"},
        {"event": "segment_finalized", "t_ms": 5000.0, "duration_s": 2.0, "reason": "silence"},
        {"event": "translate", "seq": 1, "jp": "こんにちは", "vi": "Xin chào", "latency_ms": 600.0},
        {"event": "display", "seq": 1, "jp": "こんにちは", "vi": "Xin chào", "t_ms": 5600.0, "ts": "10:00:05"},
        {"event": "segment_finalized", "t_ms": 9000.0, "duration_s": 3.0, "reason": "max_utterance"},
        {"event": "translate", "seq": 2, "jp": "元気ですか", "vi": "Bạn khỏe không", "latency_ms": 800.0},
        {"event": "display", "seq": 2, "jp": "元気ですか", "vi": "Bạn khỏe không", "t_ms": 9800.0, "ts": "10:00:09"},
    ]


def test_build_lines_pairs_and_timing() -> None:
    lines = te.build_lines(_sample_events())
    assert len(lines) == 2
    assert lines[0].jp == "こんにちは" and lines[0].vi == "Xin chào"
    # start = end(5000) - duration(2000) = 3000ms ; end = 5000ms.
    assert lines[0].start_ms == 3000.0 and lines[0].end_ms == 5000.0
    assert lines[1].start_ms == 6000.0 and lines[1].end_ms == 9000.0
    print("test_build_lines_pairs_and_timing OK")


def test_srt_format() -> None:
    lines = te.build_lines(_sample_events())
    srt = te.to_srt(lines)
    assert "1\n00:00:03,000 --> 00:00:05,000" in srt
    assert "Xin chào" in srt and "こんにちは" in srt
    # Second cue index + bilingual body present.
    assert "2\n00:00:06,000 --> 00:00:09,000" in srt
    print("test_srt_format OK")


def test_partial_log_zips_to_shorter() -> None:
    # display without a matching segment_finalized still renders (live capture).
    events = [
        {"event": "display", "seq": 1, "jp": "あ", "vi": "A", "t_ms": 1000.0, "ts": "x"},
    ]
    lines = te.build_lines(events)
    assert len(lines) == 1 and lines[0].start_ms == lines[0].end_ms == 1000.0
    print("test_partial_log_zips_to_shorter OK")


def test_summarize_counts_reasons_and_loss() -> None:
    events = _sample_events() + [{"event": "empty_translation", "seq": 3}]
    stats = te.summarize(events)
    assert stats["segments"] == 2 and stats["displayed"] == 2
    assert stats["reasons"] == {"silence": 1, "max_utterance": 1}
    assert stats["max_utterance_pct"] == 50.0
    assert stats["loss_events"] == {"empty_translation": 1}
    print("test_summarize_counts_reasons_and_loss OK")


def test_text_and_json_render() -> None:
    lines = te.build_lines(_sample_events())
    txt = te.to_text(lines)
    assert "JP こんにちは" in txt and "VI Xin chào" in txt
    js = te.to_json(lines)
    assert '"seq": 1' in js and "Bạn khỏe không" in js
    print("test_text_and_json_render OK")


def main() -> None:
    test_build_lines_pairs_and_timing()
    test_srt_format()
    test_partial_log_zips_to_shorter()
    test_summarize_counts_reasons_and_loss()
    test_text_and_json_render()
    print("All transcript_export tests passed.")


if __name__ == "__main__":
    main()
