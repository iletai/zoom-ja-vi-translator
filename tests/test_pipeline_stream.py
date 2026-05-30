"""End-to-end streaming pipeline test that simulates a live Zoom meeting.

Instead of capturing from a physical loopback device (which is unavailable in
CI / headless runs), a FakeCapture thread streams real Japanese speech wavs into
the pipeline's audio queue in 0.2 s blocks — exactly the shape AudioCapture
produces. This exercises the real VAD -> ASR -> translate -> display flow across
the three pipeline threads, queues, drop-oldest logic and graceful shutdown.
"""
from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.display import SubtitleDisplay  # noqa: E402
from src.pipeline import TranslationPipeline  # noqa: E402

WAV_DIR = ROOT / "test_audio" / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" / "test_wavs"
BLOCK_SECONDS = config.CAPTURE_BLOCK_SECONDS


def load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class RecordingDisplay(SubtitleDisplay):
    """Capture subtitle pairs while still printing them.

    The pipeline now prints the Japanese line as soon as ASR completes
    (``show_source``) and the Vietnamese line when translation finishes
    (``show_target``), so we pair the most recent source with its target.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[tuple[str, str]] = []
        self._pending_ja: str | None = None

    def show(self, japanese: str, vietnamese: str) -> None:
        self.pairs.append((japanese, vietnamese))
        super().show(japanese, vietnamese)

    def show_source(self, japanese: str) -> None:
        self._pending_ja = japanese
        super().show_source(japanese)

    def show_target(self, vietnamese: str) -> None:
        if self._pending_ja is not None:
            self.pairs.append((self._pending_ja, vietnamese))
            self._pending_ja = None
        super().show_target(vietnamese)


class FakeCapture(threading.Thread):
    """Stream wav audio into the pipeline queue like the real AudioCapture."""

    def __init__(self, audio: np.ndarray, output_queue, stop_event, realtime: bool) -> None:
        super().__init__(daemon=True)
        self.audio = audio
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.realtime = realtime
        self.error = None

    def run(self) -> None:
        block = int(config.SAMPLE_RATE * BLOCK_SECONDS)
        for off in range(0, self.audio.size, block):
            if self.stop_event.is_set():
                return
            chunk = self.audio[off : off + block].astype(np.float32, copy=False)
            try:
                self.output_queue.put(chunk, timeout=1.0)
            except Exception:
                pass
            if self.realtime:
                time.sleep(BLOCK_SECONDS)
        # Give the ASR/translate stages time to drain before we let main stop.
        self.stop_event.wait(0.1)


def build_stream() -> np.ndarray:
    """Concatenate a few wavs with 0.7 s silence gaps to mimic meeting turns."""
    wavs = sorted(WAV_DIR.glob("*.wav"))[:3]
    gap = np.zeros(int(config.SAMPLE_RATE * 0.7), dtype=np.float32)
    parts: list[np.ndarray] = []
    for wav in wavs:
        parts.append(load_wav_float32(wav))
        parts.append(gap)
    return np.concatenate(parts)


def main() -> int:
    realtime = "--fast" not in sys.argv
    stream = build_stream()
    duration = stream.size / config.SAMPLE_RATE
    print(f"Simulated meeting audio: {duration:.1f}s "
          f"({'real-time' if realtime else 'fast'} mode)\n")

    display = RecordingDisplay()
    pipeline = TranslationPipeline(device=None, display=display)

    # Swap the real loopback capture for the wav streamer.
    pipeline._capture = FakeCapture(
        stream, pipeline._audio_queue, pipeline.stop_event, realtime=realtime
    )

    pipeline.start()

    # Wait until the fake capture has streamed everything, then allow drain time.
    pipeline._capture.join()
    deadline = time.time() + 60
    expected = 3
    while time.time() < deadline:
        if len(display.pairs) >= expected and pipeline._audio_queue.empty():
            time.sleep(2.0)  # let any final utterance flush through
            break
        time.sleep(0.5)

    pipeline.stop()

    print(f"\n=== Captured {len(display.pairs)} subtitle pair(s) ===")
    for ja, vi in display.pairs:
        print(f"  JA: {ja}")
        print(f"  VI: {vi}")

    ok = (
        len(display.pairs) >= 2
        and all(ja and vi for ja, vi in display.pairs)
    )
    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} "
          f"({len(display.pairs)} pairs) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
