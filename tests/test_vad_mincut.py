"""Unit tests for max-utterance min-cut and optional VAD energy gate.

The segmenter is driven by a scripted fake VAD so tests are deterministic and do
not rely on real speech. Each audio frame is one VAD frame; the fake returns the
next scripted voiced/unvoiced flag per frame.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from src import vad as vad_mod  # noqa: E402


class _ScriptedVad:
    def __init__(self, flags: list[bool]) -> None:
        self.flags = list(flags)
        self.idx = 0

    def is_speech(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        if self.idx >= len(self.flags):
            return self.flags[-1] if self.flags else False
        result = self.flags[self.idx]
        self.idx += 1
        return result


class _FakeWebrtcVad:
    class Vad:
        def __init__(self, aggressiveness: int) -> None:
            self.aggressiveness = aggressiveness

        def is_speech(self, pcm_bytes: bytes, sample_rate: int) -> bool:
            return False


def _frames(values: list[float]) -> np.ndarray:
    frame_samples = int(config.SAMPLE_RATE) * int(config.VAD_FRAME_MS) // 1000
    parts = [np.full(frame_samples, value, dtype=np.float32) for value in values]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


class _ConfigPatch:
    def __init__(self, **updates) -> None:
        self.updates = updates
        self.originals = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.originals[key] = getattr(config, key)
            setattr(config, key, value)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.originals.items():
            setattr(config, key, value)


def _make_segmenter(flags: list[bool], **overrides) -> "vad_mod.VadSegmenter":
    defaults = {
        "VAD_PREROLL_MS": 0,
        "VAD_MIN_UTTERANCE_MS": 30,
        "VAD_SILENCE_MS": 300,
        "VAD_MAX_UTTERANCE_MS": 360,
        "VAD_ENERGY_GATE": False,
    }
    defaults.update(overrides)
    orig_webrtcvad = vad_mod.webrtcvad
    if vad_mod.webrtcvad is None:
        vad_mod.webrtcvad = _FakeWebrtcVad
    try:
        with _ConfigPatch(**defaults):
            seg = vad_mod.VadSegmenter()
    finally:
        vad_mod.webrtcvad = orig_webrtcvad
    seg._vad = _ScriptedVad(flags)
    return seg


def test_mincut_splits_at_internal_silence_and_carries_tail() -> None:
    first_values = [0.5] * 12 + [0.0] * 3 + [0.8] * 10
    first_flags = [True] * 12 + [False] * 3 + [True] * 10
    more_values = [0.6] * 4 + [0.0] * 10
    more_flags = [True] * 4 + [False] * 10
    frame_samples = int(config.SAMPLE_RATE) * int(config.VAD_FRAME_MS) // 1000

    seg = _make_segmenter(first_flags + more_flags, VAD_MAX_UTTERANCE_MS=750)
    first = seg.push(_frames(first_values))
    assert len(first) == 1, first
    assert len(first[0]) <= 12 * frame_samples

    second = seg.push(_frames(more_values))
    assert len(second) == 1, second
    carried_tail = 10 * frame_samples
    assert np.allclose(second[0][:carried_tail], 0.8)

    total_out = len(first[0]) + len(second[0])
    dropped_gap = 3 * frame_samples
    total_in = (len(first_values) + len(more_values)) * frame_samples
    assert total_out == total_in - dropped_gap, (total_out, total_in - dropped_gap)


def test_mincut_fallback_rms_emits_and_carries() -> None:
    values = [0.5] * 5 + [0.4, 0.3, 0.05, 0.35, 0.45]
    seg = _make_segmenter([True] * len(values), VAD_MAX_UTTERANCE_MS=300)

    emitted = seg.push(_frames(values))

    assert len(emitted) == 1, emitted
    assert emitted[0].size > 0
    assert 0 < len(seg._utterance_frames) < len(values)


def test_energy_gate_default_off_keeps_webrtc_voiced_run_open() -> None:
    values = [0.1] * 4 + [0.001] * 8
    seg = _make_segmenter(
        [True] * len(values),
        VAD_SILENCE_MS=90,
        VAD_MAX_UTTERANCE_MS=3000,
        VAD_ENERGY_GATE=False,
    )

    emitted = seg.push(_frames(values))

    assert emitted == []
    assert seg._current_silence_ms == 0


def test_energy_gate_on_can_endpoint_low_rms_webrtc_voiced_tail() -> None:
    values = [0.001] * 3 + [0.1] * 4 + [0.001] * 4
    flags = [False] * 3 + [True] * 8
    seg = _make_segmenter(
        flags,
        VAD_SILENCE_MS=90,
        VAD_MAX_UTTERANCE_MS=3000,
        VAD_ENERGY_GATE=True,
    )

    emitted = seg.push(_frames(values))

    assert len(emitted) == 1, emitted
    assert seg._current_silence_ms == 0


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
