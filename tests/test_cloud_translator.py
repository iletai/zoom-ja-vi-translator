"""Unit tests for the Azure cloud backend wiring.

These run WITHOUT an API key or network: they exercise the pure logic
(float32 -> PCM conversion and the recognizing/recognized -> display mapping)
using a fake recognizer/event objects. The Azure SDK is never imported.

Run from the project root:

    python3 tests/test_cloud_translator.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.cloud_translator import AzureSpeechTranslator, float32_to_pcm16  # noqa: E402
from src.display import SubtitleDisplay  # noqa: E402


class RecordingDisplay(SubtitleDisplay):
    """Capture display calls instead of (only) printing them."""

    def __init__(self) -> None:
        super().__init__()
        self.partials: list[str] = []
        self.finals: list[str] = []
        self.targets: list[str] = []

    def show_source_partial(self, japanese: str) -> None:
        self.partials.append(japanese)

    def finalize_source(self, japanese: str) -> None:
        self.finals.append(japanese)

    def show_target(self, vietnamese: str) -> None:
        self.targets.append(vietnamese)


class FakeResult:
    def __init__(self, text: str, translations: dict | None = None) -> None:
        self.text = text
        self.translations = translations or {}


class FakeEvt:
    def __init__(self, result: FakeResult) -> None:
        self.result = result


def _new_translator(display: SubtitleDisplay) -> AzureSpeechTranslator:
    # Provide dummy credentials so construction succeeds without env vars; the
    # SDK is never built in these tests (we call the handlers directly).
    return AzureSpeechTranslator(
        display, key="dummy-key", region="dummy-region",
    )


def test_pcm_conversion() -> None:
    # 0.0 -> 0, 1.0 -> 32767, -1.0 -> -32767, and clipping beyond +-1.
    samples = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = float32_to_pcm16(samples)
    values = struct.unpack("<5h", pcm)
    assert values == (0, 32767, -32767, 32767, -32767), values
    # Empty / None inputs are safe.
    assert float32_to_pcm16(np.array([], dtype=np.float32)) == b""
    assert float32_to_pcm16(None) == b""
    # 16-bit => 2 bytes per sample.
    assert len(float32_to_pcm16(np.zeros(10, dtype=np.float32))) == 20
    print("[pcm] OK")


def test_partial_dedup() -> None:
    display = RecordingDisplay()
    t = _new_translator(display)
    t._on_recognizing(FakeEvt(FakeResult("こんに")))
    t._on_recognizing(FakeEvt(FakeResult("こんに")))   # duplicate -> ignored
    t._on_recognizing(FakeEvt(FakeResult("こんにちは")))
    t._on_recognizing(FakeEvt(FakeResult("   ")))       # blank -> ignored
    assert display.partials == ["こんに", "こんにちは"], display.partials
    print("[partial] OK")


def test_final_shows_source_and_target() -> None:
    display = RecordingDisplay()
    t = _new_translator(display)
    t._on_recognizing(FakeEvt(FakeResult("おはよ")))
    evt = FakeEvt(FakeResult("おはようございます", {config.CLOUD_TARGET_LANG: "Chào buổi sáng"}))
    t._on_recognized(evt)
    assert display.finals == ["おはようございます"], display.finals
    assert display.targets == ["Chào buổi sáng"], display.targets
    # After a final, the partial state resets so the next utterance re-shows.
    assert t._last_partial == ""
    print("[final] OK")


def test_final_without_translation_uses_placeholder() -> None:
    display = RecordingDisplay()
    t = _new_translator(display)
    t._on_recognized(FakeEvt(FakeResult("テスト", {})))   # no translation present
    assert display.finals == ["テスト"], display.finals
    assert display.targets == ["(...)"], display.targets
    print("[final-placeholder] OK")


def test_empty_final_ignored() -> None:
    display = RecordingDisplay()
    t = _new_translator(display)
    t._on_recognized(FakeEvt(FakeResult("   ", {config.CLOUD_TARGET_LANG: "x"})))
    assert display.finals == [] and display.targets == []
    print("[empty-final] OK")


def test_missing_credentials_raises() -> None:
    raised = False
    try:
        AzureSpeechTranslator(SubtitleDisplay(), key="", region="")
    except ValueError:
        raised = True
    assert raised, "expected ValueError when credentials are missing"
    print("[creds] OK")


class FakeCancelEvt:
    def __init__(self, error_details: str = "", reason: str = "") -> None:
        self.error_details = error_details
        self.reason = reason


def test_canceled_triggers_on_fatal() -> None:
    display = RecordingDisplay()
    fired: list[str] = []
    t = AzureSpeechTranslator(
        display, key="k", region="r", on_fatal=lambda msg: fired.append(msg)
    )
    t._on_canceled(FakeCancelEvt(error_details="auth failed"))
    assert len(fired) == 1 and "auth failed" in fired[0], fired
    print("[canceled-fatal] OK")


def test_callbacks_suppressed_after_stopping() -> None:
    display = RecordingDisplay()
    fired: list[str] = []
    t = AzureSpeechTranslator(
        display, key="k", region="r", on_fatal=lambda msg: fired.append(msg)
    )
    t._stopping = True  # simulate shutdown in progress
    t._on_recognizing(FakeEvt(FakeResult("無視")))
    t._on_recognized(FakeEvt(FakeResult("無視", {config.CLOUD_TARGET_LANG: "x"})))
    t._on_canceled(FakeCancelEvt(error_details="expected during stop"))
    assert display.partials == [] and display.finals == [] and display.targets == []
    assert fired == [], "fatal callback must not fire during intentional stop"
    print("[suppress-after-stop] OK")


def main() -> int:
    test_pcm_conversion()
    test_partial_dedup()
    test_final_shows_source_and_target()
    test_final_without_translation_uses_placeholder()
    test_empty_final_ignored()
    test_missing_credentials_raises()
    test_canceled_triggers_on_fatal()
    test_callbacks_suppressed_after_stopping()
    print("\n=== RESULT: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
