"""Script-style tests for empty translation attribution in batched display."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import evidence_log as ev  # noqa: E402
from src.pipeline import TranslationPipeline  # noqa: E402
from src.translator import join_translations  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "_artifacts"


class SomeEmptyTranslator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, text: str) -> str:
        return "" if "二つ目" in text else f"VI({text})"

    def translate_many(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        return [self.translate(text) for text in texts]


class RecordingDisplay:
    def __init__(self) -> None:
        self.targets: list[tuple[int | None, str, str | None]] = []
        self.pairs: list[tuple[str, str]] = []
        self.infos: list[str] = []

    def show_target(self, vietnamese: str, japanese=None, seq=None) -> None:
        self.targets.append((seq, vietnamese, japanese))

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self.pairs.append((japanese, vietnamese))

    def info(self, message: str) -> None:
        self.infos.append(message)


def make_stub(log_path: Path) -> TranslationPipeline:
    ev.close()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    ev.configure(str(log_path))

    inst = TranslationPipeline.__new__(TranslationPipeline)
    inst._text_queue = queue.Queue(maxsize=32)
    inst._seq = 0
    inst._seq_lock = threading.Lock()
    inst.stop_event = threading.Event()
    inst._draining = threading.Event()
    inst._last_enqueued_text = ""
    inst._last_enqueued_at = 0.0
    inst.translator = SomeEmptyTranslator()
    inst.display = RecordingDisplay()
    return inst


def read_events(path: Path) -> list[dict]:
    ev.close()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_join_translations_keeps_empty_placeholder() -> None:
    joined, dropped = join_translations(["一つ目です。", "二つ目です。"], ["Một", ""])
    assert joined == "Một (...)", joined
    assert dropped == [(1, "二つ目です。")], dropped


def test_batch_empty_translation_attribution() -> None:
    log_path = ARTIFACT_DIR / "join_translations_empty.jsonl"
    inst = make_stub(log_path)
    batch = [
        (1, "一つ目です。二つ目です。", True),
        (2, "三つ目です。", True),
    ]

    TranslationPipeline._translate_and_display_batch(inst, batch)
    events = read_events(log_path)

    targets = {seq: (vi, jp) for seq, vi, jp in inst.display.targets}
    assert set(targets) == {1, 2}, targets
    assert targets[1][1] == "一つ目です。二つ目です。", targets[1]
    assert "VI(一つ目です。)" in targets[1][0], targets[1]
    assert "(...)" in targets[1][0], targets[1]
    assert "三つ目" not in targets[1][0], targets[1]
    assert targets[2][1] == "三つ目です。", targets[2]
    assert targets[2][0] == "VI(三つ目です。)", targets[2]

    empty_events = [event for event in events if event["event"] == "empty_translation"]
    assert len(empty_events) == 1, empty_events
    assert empty_events[0]["seq"] == 1, empty_events[0]
    assert empty_events[0]["sentence_index"] == 1, empty_events[0]
    assert empty_events[0]["jp"] == "二つ目です。", empty_events[0]


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
    ev.close()
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
