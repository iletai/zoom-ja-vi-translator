"""Script-style robustness tests for the batched translate/display path.

These cover the silent-data-loss failure modes found in the code review:
per-item attribution across a flattened batch (B1), translation/source count
mismatch padding, display-failure isolation, shutdown queue draining, and
backpressure never silently dropping recognized text (B2).

No model load: a faithful in-memory translator/display stub is used so the
tests exercise pipeline wiring, not NLLB.
"""
from __future__ import annotations

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.pipeline import TranslationPipeline  # noqa: E402


class StubTranslator:
    """One-to-one fake: translate_many mirrors translate so attribution holds."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, text: str) -> str:
        return f"VI({text})"

    def translate_many(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        return [self.translate(t) for t in texts]


class ShortTranslator(StubTranslator):
    """Returns FEWER translations than sentences to trigger the count guard."""

    def translate_many(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        return [self.translate(t) for t in texts][:-1] if texts else []


class RecordingDisplay:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.targets: list[tuple[str, str]] = []
        self.infos: list[str] = []

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self.pairs.append((japanese, vietnamese))

    def show_target(self, vietnamese: str, japanese=None, seq=None) -> None:
        self.targets.append((japanese, vietnamese))

    def info(self, message: str) -> None:
        self.infos.append(message)


class FailingDisplay(RecordingDisplay):
    """show_pair raises for one specific JP so we can prove isolation."""

    def __init__(self, fail_jp: str) -> None:
        super().__init__()
        self._fail_jp = fail_jp

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        if japanese == self._fail_jp:
            raise RuntimeError("render boom")
        super().show_pair(japanese, vietnamese)


def make_stub(translator=None, display=None, maxsize=256) -> TranslationPipeline:
    inst = TranslationPipeline.__new__(TranslationPipeline)
    inst._text_queue = queue.Queue(maxsize=maxsize)
    inst._seq = 0
    inst._seq_lock = threading.Lock()
    inst.stop_event = threading.Event()
    inst._draining = threading.Event()
    inst._translate_thread = None
    inst._webhook_pool = None
    inst._last_enqueued_text = ""
    inst._last_enqueued_at = 0.0
    inst.translator = translator or StubTranslator()
    inst.display = display or RecordingDisplay()
    return inst


def test_attribution_across_differing_sentence_counts() -> None:
    """B1: a flattened batch must rejoin per item even when items differ in
    sentence count (cursor must not slide between items)."""
    inst = make_stub()
    multi = "今日は晴れです。明日は雨です。"  # 2 sentences
    single = "ありがとうございます"            # 1 sentence
    batch = [(1, multi, False), (2, single, False)]

    TranslationPipeline._translate_and_display_batch(inst, batch)

    pairs = dict((jp, vi) for jp, vi in inst.display.pairs)
    assert len(inst.display.pairs) == 2, inst.display.pairs
    # The multi-sentence item must carry BOTH sentence translations (no drop).
    assert "今日は晴れです" in pairs[multi] and "明日は雨です" in pairs[multi], pairs[multi]
    # The single item must NOT absorb the other item's sentences.
    assert "今日" not in pairs[single] and "明日" not in pairs[single], pairs[single]


def test_count_mismatch_pads_and_never_truncates() -> None:
    """A short translation list must pad (show '(...)') not zip-truncate the
    unpaired tail item out of existence."""
    inst = make_stub(translator=ShortTranslator())
    batch = [(1, "おはよう", False), (2, "こんばんは", False)]

    TranslationPipeline._translate_and_display_batch(inst, batch)

    # Both items still rendered; the unpaired one shows the placeholder.
    assert len(inst.display.pairs) == 2, inst.display.pairs
    jps = [jp for jp, _ in inst.display.pairs]
    assert "おはよう" in jps and "こんばんは" in jps


def test_display_failure_does_not_drop_other_items() -> None:
    """A render exception on one item must not abort the rest of the batch."""
    inst = make_stub(display=None)
    inst.display = FailingDisplay(fail_jp="ふたつめ")
    batch = [(1, "ひとつめ", False), (2, "ふたつめ", False), (3, "みっつめ", False)]

    TranslationPipeline._translate_and_display_batch(inst, batch)

    shown = [jp for jp, _ in inst.display.pairs]
    assert "ひとつめ" in shown and "みっつめ" in shown, shown
    assert "ふたつめ" not in shown  # the one that raised


def test_drain_flushes_all_queued_items_across_batches() -> None:
    """stop()'s drain must translate/display every queued item, in multiple
    TRANSLATE_MAX_BATCH chunks if needed (no leftover loss)."""
    inst = make_stub()
    total = config.TRANSLATE_MAX_BATCH + 3
    for i in range(total):
        inst._text_queue.put((i, f"文{i}です", False))
    # Allow the queue to hold them all for this test.
    assert inst._text_queue.qsize() == total

    TranslationPipeline._drain_text_queue(inst)

    assert inst._text_queue.empty()
    assert len(inst.display.pairs) == total, len(inst.display.pairs)


def test_backpressure_blocks_then_abandons_without_dropping_silently() -> None:
    """B2: when the queue is full, _enqueue_text must block (not drop). When
    stop_event fires it must return (abandoning that one in-flight item, which
    is evidence-logged) rather than dropping already-queued text."""
    inst = make_stub(maxsize=4)
    # Fill the queue to capacity (maxsize=4).
    for i in range(4):
        inst._text_queue.put((i, f"既存{i}", False))
    assert inst._text_queue.full()

    done = threading.Event()

    def producer() -> None:
        TranslationPipeline._enqueue_text(inst, "あたらしい文です", pre_shown=False)
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    # It must be blocked (queue is full), not silently dropping.
    assert not done.wait(0.3), "enqueue should block on a full queue, not drop"

    # Signal stop: the blocked enqueue must unblock and return.
    inst.stop_event.set()
    assert done.wait(2.0), "enqueue did not return after stop_event"

    # None of the already-queued items were lost.
    assert inst._text_queue.qsize() == 4


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
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} "
          f"({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
