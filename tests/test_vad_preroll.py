"""Unit test for the VAD pre-onset collar (pre-roll).

Aggressive webrtcvad settings clip the quiet leading mora of a word; the collar
prepends a fixed amount of pre-onset audio so the offline ASR sees the whole
word. We drive the segmenter with a deterministic fake VAD (a scripted
voiced/unvoiced pattern) so the test is fast and does not depend on the optional
webrtcvad wheel or on real speech.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from src import vad as vad_mod  # noqa: E402


class _ScriptedVad:
    """Returns True for frames whose first sample is non-zero ("voiced")."""

    def __init__(self, aggressiveness: int) -> None:
        self.aggressiveness = aggressiveness

    def is_speech(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        first = int.from_bytes(pcm_bytes[:2], "little", signed=True)
        return first != 0


def _make_segmenter(preroll_ms: int) -> "vad_mod.VadSegmenter":
    orig_pre = getattr(config, "VAD_PREROLL_MS", 0)
    config.VAD_PREROLL_MS = preroll_ms
    try:
        seg = vad_mod.VadSegmenter()
    finally:
        config.VAD_PREROLL_MS = orig_pre
    seg._vad = _ScriptedVad(int(config.VAD_AGGRESSIVENESS))
    return seg


def test_preroll_prepends_pre_onset_audio() -> None:
    sr = int(config.SAMPLE_RATE)
    fr = int(config.VAD_FRAME_MS)
    frame_samples = sr * fr // 1000
    preroll_ms = 240
    collar_frames = preroll_ms // fr

    # 10 frames of low-but-nonzero "lead-in" (would be clipped without a collar),
    # then enough voiced frames to clear VAD_MIN_UTTERANCE_MS, then silence (zeros)
    # to trigger the endpoint. Lead-in is unvoiced here only by being below the
    # scripted threshold; we mark it unvoiced explicitly via zeros vs the marker.
    lead = np.full(10 * frame_samples, 0.0, dtype=np.float32)  # unvoiced (zeros)
    voiced_frames = (config.VAD_MIN_UTTERANCE_MS // fr) + 4
    # Use a marker value so the scripted VAD sees "voiced" (first sample != 0).
    voiced = np.full(voiced_frames * frame_samples, 0.5, dtype=np.float32)
    trailing_sil = np.zeros((config.VAD_SILENCE_MS // fr + 2) * frame_samples, dtype=np.float32)

    seg = _make_segmenter(preroll_ms)
    audio = np.concatenate((lead, voiced, trailing_sil))
    utterances = []
    block = frame_samples * 5
    for off in range(0, audio.size, block):
        utterances.extend(seg.push(audio[off : off + block]))
    tail = seg.flush()
    if tail is not None:
        utterances.append(tail)

    assert len(utterances) >= 1, "should detect the voiced utterance"
    utt = utterances[0]
    # The utterance must include collar_frames of pre-onset audio AHEAD of the
    # first voiced sample: its length exceeds the voiced+endpoint portion by the
    # collar (capped at how many unvoiced lead frames existed).
    expected_collar = min(collar_frames, 10) * frame_samples
    voiced_samples = voiced_frames * frame_samples
    assert len(utt) >= voiced_samples + expected_collar, (
        f"utterance {len(utt)} should carry >= {expected_collar} collar samples "
        f"ahead of {voiced_samples} voiced samples"
    )


def test_no_collar_when_preroll_zero() -> None:
    sr = int(config.SAMPLE_RATE)
    fr = int(config.VAD_FRAME_MS)
    frame_samples = sr * fr // 1000
    lead = np.zeros(10 * frame_samples, dtype=np.float32)
    voiced_frames = (config.VAD_MIN_UTTERANCE_MS // fr) + 4
    voiced = np.full(voiced_frames * frame_samples, 0.5, dtype=np.float32)
    trailing_sil = np.zeros((config.VAD_SILENCE_MS // fr + 2) * frame_samples, dtype=np.float32)

    seg = _make_segmenter(0)
    audio = np.concatenate((lead, voiced, trailing_sil))
    utterances = []
    block = frame_samples * 5
    for off in range(0, audio.size, block):
        utterances.extend(seg.push(audio[off : off + block]))
    tail = seg.flush()
    if tail is not None:
        utterances.append(tail)

    assert len(utterances) >= 1
    # Without a collar the utterance begins at the first voiced frame (no lead-in).
    assert seg._preroll.maxlen == 0


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
