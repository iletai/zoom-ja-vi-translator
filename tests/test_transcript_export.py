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
        {"event": "segment_finalized", "seq": 1, "t_ms": 5000.0, "duration_s": 2.0, "reason": "silence"},
        {"event": "translate", "seq": 1, "jp": "こんにちは", "vi": "Xin chào", "latency_ms": 600.0},
        {"event": "display", "seq": 1, "jp": "こんにちは", "vi": "Xin chào", "t_ms": 5600.0, "ts": "10:00:05"},
        {"event": "segment_finalized", "seq": 2, "t_ms": 9000.0, "duration_s": 3.0, "reason": "max_utterance"},
        {"event": "translate", "seq": 2, "jp": "元気ですか", "vi": "Bạn khỏe không", "latency_ms": 800.0},
        {"event": "display", "seq": 2, "jp": "元気ですか", "vi": "Bạn khỏe không", "t_ms": 9800.0, "ts": "10:00:09"},
    ]


def test_build_lines_pairs_and_timing() -> None:
    lines = te.build_lines(_sample_events())
    assert len(lines) == 2
    assert lines[0].jp == "こんにちは" and lines[0].vi == "Xin chào"
    # Timing is derived from display t_ms: line spans to the next display's t_ms.
    # line0: 5600 -> 9800 (gap 4200, within [800, 8000]).
    assert lines[0].start_ms == 5600.0 and lines[0].end_ms == 9800.0
    # line1: last line gets the default 3000ms span from its own t_ms (9800).
    assert lines[1].start_ms == 9800.0 and lines[1].end_ms == 12800.0
    print("test_build_lines_pairs_and_timing OK")


def test_srt_format() -> None:
    lines = te.build_lines(_sample_events())
    srt = te.to_srt(lines)
    assert "1\n00:00:05,600 --> 00:00:09,800" in srt
    assert "Xin chào" in srt and "こんにちは" in srt
    # Second cue index + bilingual body present.
    assert "2\n00:00:09,800 --> 00:00:12,800" in srt
    print("test_srt_format OK")


def test_single_display_line_gets_default_duration() -> None:
    # A lone display (live capture / last line) still renders with a sane span.
    events = [
        {"event": "display", "seq": 1, "jp": "あ", "vi": "A", "t_ms": 1000.0, "ts": "x"},
    ]
    lines = te.build_lines(events)
    assert len(lines) == 1
    assert lines[0].start_ms == 1000.0 and lines[0].end_ms == 4000.0  # +3000 default
    print("test_single_display_line_gets_default_duration OK")


def test_skipped_display_does_not_shift_timing() -> None:
    # A finalized segment that produces NO display (empty_translation) must not
    # shift the timing of later lines: each line's timing comes from its OWN
    # display t_ms, never from a positional/seq join to segment_finalized.
    events = [
        {"event": "segment_finalized", "seq": 1, "t_ms": 5000.0, "duration_s": 2.0, "reason": "silence"},
        {"event": "display", "seq": 1, "jp": "一", "vi": "Một", "t_ms": 5100.0, "ts": "10:00:05"},
        {"event": "segment_finalized", "seq": 2, "t_ms": 8000.0, "duration_s": 1.5, "reason": "silence"},
        {"event": "empty_translation", "seq": 2},
        {"event": "segment_finalized", "seq": 3, "t_ms": 12000.0, "duration_s": 2.0, "reason": "silence"},
        {"event": "display", "seq": 3, "jp": "三", "vi": "Ba", "t_ms": 12100.0, "ts": "10:00:12"},
    ]
    lines = te.build_lines(events)
    assert [ln.seq for ln in lines] == [1, 3]
    # seq3 keeps its OWN display time (12100), not shifted by the skipped seq2.
    assert lines[1].start_ms == 12100.0
    assert lines[0].start_ms == 5100.0 and lines[0].end_ms == 12100.0
    print("test_skipped_display_does_not_shift_timing OK")


def test_build_lines_orders_by_display_time() -> None:
    # Displays arriving out of order are sorted onto a monotonic timeline.
    events = [
        {"event": "display", "seq": 2, "jp": "二", "vi": "Hai", "t_ms": 4600.0, "ts": "10:00:04"},
        {"event": "display", "seq": 1, "jp": "一", "vi": "Một", "t_ms": 3100.0, "ts": "10:00:03"},
    ]
    lines = te.build_lines(events)
    assert [ln.seq for ln in lines] == [1, 2]
    assert all(lines[i].start_ms <= lines[i + 1].start_ms for i in range(len(lines) - 1))
    srt = te.to_srt(lines)
    assert "1\n00:00:03,100 --> 00:00:04,600" in srt
    assert "2\n00:00:04,600 --> 00:00:07,600" in srt
    print("test_build_lines_orders_by_display_time OK")


def test_duplicate_seq_collapses_to_last_display() -> None:
    # A pre-shown line later superseded by its final text collapses to the last.
    events = [
        {"event": "display", "seq": 1, "jp": "あ", "vi": "tam", "t_ms": 1000.0, "ts": "x", "pre_shown": True},
        {"event": "display", "seq": 1, "jp": "ありがとう", "vi": "Cảm ơn", "t_ms": 1200.0, "ts": "x"},
    ]
    lines = te.build_lines(events)
    assert len(lines) == 1
    assert lines[0].vi == "Cảm ơn" and lines[0].jp == "ありがとう"
    print("test_duplicate_seq_collapses_to_last_display OK")


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
    test_single_display_line_gets_default_duration()
    test_skipped_display_does_not_shift_timing()
    test_build_lines_orders_by_display_time()
    test_duplicate_seq_collapses_to_last_display()
    test_summarize_counts_reasons_and_loss()
    test_text_and_json_render()
    print("All transcript_export tests passed.")


if __name__ == "__main__":
    main()
