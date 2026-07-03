"""Script-style shutdown audit tests for no silent recognized-text loss."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src import evidence_log as ev  # noqa: E402
from src.pipeline import TranslationPipeline  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "_artifacts"


class EchoTranslator:
    def translate(self, text: str) -> str:
        return f"VI({text})"

    def translate_many(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]


class RecordingDisplay:
    def __init__(self) -> None:
        self.sources: list[tuple[int | None, str]] = []
        self.targets: list[tuple[int | None, str, str | None]] = []
        self.pairs: list[tuple[str, str]] = []
        self.infos: list[str] = []

    def show_source(self, japanese: str, seq=None) -> None:
        self.sources.append((seq, japanese))

    def show_target(self, vietnamese: str, japanese=None, seq=None) -> None:
        self.targets.append((seq, vietnamese, japanese))

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self.pairs.append((japanese, vietnamese))

    def info(self, message: str) -> None:
        self.infos.append(message)


class FakeCapture(threading.Thread):
    def __init__(self, utterances: list[str], output_queue, stop_event) -> None:
        super().__init__(daemon=True)
        self.utterances = utterances
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.error = None

    def run(self) -> None:
        for utterance in self.utterances:
            if self.stop_event.is_set():
                return
            self.output_queue.put(utterance)


class FakeASR:
    def transcribe(self, utterance: str) -> str:
        return utterance


class FakeSegmenter:
    def push(self, block: str) -> list[str]:
        return [block]

    def flush(self):
        return None


class JoinableNoop:
    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class BlockingThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.release = threading.Event()

    def run(self) -> None:
        self.release.wait(5.0)


def configure_log(name: str) -> Path:
    ev.close()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    path = ARTIFACT_DIR / name
    if path.exists():
        path.unlink()
    ev.configure(str(path))
    return path


def read_events(path: Path) -> list[dict]:
    ev.close()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_pipeline(utterances: list[str], log_name: str) -> tuple[TranslationPipeline, RecordingDisplay, Path]:
    log_path = configure_log(log_name)
    display = RecordingDisplay()
    inst = TranslationPipeline.__new__(TranslationPipeline)
    inst.device = None
    inst.display = display
    inst.streaming = False
    inst.backend = "local"
    inst.cloud = False
    inst.stop_event = threading.Event()
    inst._audio_queue = queue.Queue(maxsize=64)
    inst._text_queue = queue.Queue(maxsize=64)
    inst._seq = 0
    inst._seq_lock = threading.Lock()
    inst.asr = FakeASR()
    inst.segmenter = FakeSegmenter()
    inst.streaming_asr = None
    inst.aggregator = None
    inst._offline_aggregator = None
    inst.translator = EchoTranslator()
    inst._redecode = False
    inst._segment_queue = queue.Queue(maxsize=64)
    inst._redecode_thread = None
    inst._draining = threading.Event()
    inst._webhook_pool = None
    inst._last_enqueued_text = ""
    inst._last_enqueued_at = 0.0
    inst._capture = FakeCapture(utterances, inst._audio_queue, inst.stop_event)
    inst._asr_thread = threading.Thread(target=inst._asr_worker, daemon=True)
    inst._translate_thread = threading.Thread(target=inst._translate_worker, daemon=True)
    return inst, display, log_path


def enqueue_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event["event"] == "enqueue"]


def display_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event["event"] == "display"]


def assert_contiguous(seqs: list[int]) -> None:
    assert seqs == list(range(1, len(seqs) + 1)), seqs


def test_normal_stop_displays_every_enqueued_seq() -> None:
    utterances = [f"通常停止{i}です。" for i in range(1, 9)]
    pipe, display, log_path = make_pipeline(utterances, "shutdown_normal.jsonl")

    pipe.start()
    pipe._capture.join(timeout=2.0)
    deadline = time.time() + 3.0
    while time.time() < deadline and len(display.targets) < len(utterances):
        time.sleep(0.05)
    pipe.stop()
    events = read_events(log_path)

    enqueued = enqueue_events(events)
    displayed = display_events(events)
    assert len(enqueued) == len(utterances), enqueued
    assert len(displayed) == len(enqueued), displayed
    assert_contiguous([event["seq"] for event in enqueued])
    assert {event["seq"] for event in displayed} == {event["seq"] for event in enqueued}
    assert not [event for event in events if event["event"] == "shutdown_unflushed"], events
    assert len(display.targets) == len(utterances), display.targets


def test_ctrl_c_stop_audits_mid_backlog_without_seq_gaps() -> None:
    utterances = [f"割り込み{i}です。" for i in range(1, 7)]
    pipe, display, log_path = make_pipeline([], "shutdown_ctrl_c.jsonl")
    # Enqueue with no translate thread set yet: _enqueue_text treats an
    # unstarted Thread (is_alive()==False) as dead and abandons the item. In
    # production the worker is always running by enqueue time; None models that
    # "consumer alive" state so the backlog actually queues.
    pipe._translate_thread = None
    for utterance in utterances:
        TranslationPipeline._enqueue_text(pipe, utterance, pre_shown=True)

    blocker = BlockingThread()
    blocker.start()
    pipe._capture = JoinableNoop()
    pipe._asr_thread = JoinableNoop()
    pipe._translate_thread = blocker
    old_timeout = config.WORKER_SHUTDOWN_TIMEOUT
    config.WORKER_SHUTDOWN_TIMEOUT = 0.05
    try:
        pipe.stop(flush_tail=False)
    finally:
        config.WORKER_SHUTDOWN_TIMEOUT = old_timeout
        blocker.release.set()
        blocker.join(timeout=1.0)

    events = read_events(log_path)
    enqueued = enqueue_events(events)
    displayed = display_events(events)
    unflushed = [event for event in events if event["event"] == "shutdown_unflushed"]
    skipped = [event for event in events if event["event"] == "shutdown_drain_skipped"]
    queued_after_stop = [item[0] for item in list(pipe._text_queue.queue)]

    assert len(enqueued) == len(utterances), enqueued
    assert_contiguous([event["seq"] for event in enqueued])
    assert displayed == [], displayed
    assert len(unflushed) == 1, unflushed
    assert unflushed[0]["text_queue"] == len(utterances), unflushed[0]
    assert len(skipped) == 1, skipped
    assert sorted(queued_after_stop) == [event["seq"] for event in enqueued], queued_after_stop
    abandoned_count = unflushed[0]["text_queue"]
    assert len(enqueued) == len(displayed) + abandoned_count
    assert display.targets == [], display.targets


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
