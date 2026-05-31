"""Script-style tests for streaming enqueue deduplication (no model load)."""
from __future__ import annotations

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.pipeline import TranslationPipeline  # noqa: E402


def make_pipeline_stub() -> TranslationPipeline:
    inst = TranslationPipeline.__new__(TranslationPipeline)
    inst._text_queue = queue.Queue(maxsize=32)
    inst._seq = 0
    inst._seq_lock = threading.Lock()
    inst.stop_event = threading.Event()
    inst._last_enqueued_text = ""
    inst._last_enqueued_at = 0.0
    return inst


def queue_items(inst: TranslationPipeline) -> list[tuple[str, bool]]:
    # Items are (seq, japanese, pre_shown); the seq id is irrelevant to dedup.
    return [(japanese, pre_shown) for _seq, japanese, pre_shown in inst._text_queue.queue]


def test_duplicate_long_streaming_sentence_is_suppressed() -> None:
    inst = make_pipeline_stub()
    sentence = "部長三日の京都出張のスケジュールなんですが"

    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=False)
    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=False)

    assert queue_items(inst) == [(sentence, False)]


def test_short_streaming_sentence_is_not_suppressed() -> None:
    inst = make_pipeline_stub()
    short = "はい"[: max(0, config.STREAM_DEDUP_MIN_CHARS - 1)]
    assert len(short) < config.STREAM_DEDUP_MIN_CHARS

    TranslationPipeline._enqueue_text(inst, short, pre_shown=False)
    TranslationPipeline._enqueue_text(inst, short, pre_shown=False)

    assert queue_items(inst) == [(short, False), (short, False)]


def test_pre_shown_path_is_not_deduped() -> None:
    inst = make_pipeline_stub()
    sentence = "部長三日の京都出張のスケジュールなんですが"

    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=True)
    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=True)

    assert queue_items(inst) == [(sentence, True), (sentence, True)]


def test_different_streaming_sentences_are_not_suppressed() -> None:
    inst = make_pipeline_stub()
    first = "部長三日の京都出張のスケジュールなんですが"
    second = "新幹線の時間も確認しておきます"

    TranslationPipeline._enqueue_text(inst, first, pre_shown=False)
    TranslationPipeline._enqueue_text(inst, second, pre_shown=False)

    assert queue_items(inst) == [(first, False), (second, False)]


def test_duplicate_outside_window_is_not_suppressed() -> None:
    inst = make_pipeline_stub()
    sentence = "部長三日の京都出張のスケジュールなんですが"

    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=False)
    inst._last_enqueued_at -= config.STREAM_DEDUP_WINDOW_SEC + 1.0
    TranslationPipeline._enqueue_text(inst, sentence, pre_shown=False)

    assert queue_items(inst) == [(sentence, False), (sentence, False)]


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
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
