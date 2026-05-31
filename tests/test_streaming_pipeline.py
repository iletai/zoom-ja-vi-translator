"""Streaming-mode pipeline test: live partials + endpoint finalize + translate.

Exercises the real online recognizer (src/streaming_asr.py) and the streaming
pipeline worker, but stubs the translator so the test stays fast and focuses on
the low-latency ASR path. Real Japanese speech wavs are streamed in 0.2 s blocks
exactly like AudioCapture produces, with silence gaps so the recognizer's
endpoint detection fires between utterances.
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
import src.pipeline as pipeline_mod  # noqa: E402
from src.display import SubtitleDisplay  # noqa: E402

WAV_DIR = ROOT / "test_audio" / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" / "test_wavs"
BLOCK_SECONDS = config.CAPTURE_BLOCK_SECONDS


def load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class StubTranslator:
    """Instant fake translator so the test exercises wiring, not NLLB latency."""

    def translate(self, text: str) -> str:
        return f"[vi] {text[:12]}"

    def translate_many(self, texts: list[str]) -> list[str]:
        # Must mirror translate() one-to-one: the batched/flattened pipeline path
        # relies on len(out) == len(texts) for correct per-item attribution.
        return [self.translate(t) for t in texts]

    def warmup(self) -> None:
        pass


class RecordingDisplay(SubtitleDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.partials: list[str] = []
        self.finals: list[str] = []
        self.targets: list[str] = []
        self.pairs: list[tuple[str, str]] = []
        self.infos: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def show_source_partial(self, committed: str, tail: str = "") -> None:
        self.partials.append(committed + tail)

    def finalize_source(self, japanese: str) -> None:
        self.finals.append(japanese)

    def show_target(self, vietnamese: str, japanese=None, seq=None) -> None:
        self.targets.append(vietnamese)

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self.pairs.append((japanese, vietnamese))

    def show(self, japanese: str, vietnamese: str) -> None:
        self.finals.append(japanese)
        self.targets.append(vietnamese)
        self.pairs.append((japanese, vietnamese))


class FakeCapture(threading.Thread):
    def __init__(self, audio: np.ndarray, output_queue, stop_event) -> None:
        super().__init__(daemon=True)
        self.audio = audio
        self.output_queue = output_queue
        self.stop_event = stop_event
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
        self.stop_event.wait(0.1)


def build_stream() -> np.ndarray:
    """Two utterances separated by 1.3 s silence so an endpoint fires between."""
    wavs = sorted(WAV_DIR.glob("*.wav"))[:2]
    gap = np.zeros(int(config.SAMPLE_RATE * 1.3), dtype=np.float32)
    parts: list[np.ndarray] = []
    for wav in wavs:
        parts.append(load_wav_float32(wav))
        parts.append(gap)
    return np.concatenate(parts)


def has_japanese(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


def main() -> int:
    # This suite validates the legacy online-only streaming mechanics (endpoint
    # -> show_pair). The accurate offline re-decode path is covered separately in
    # test_redecode_pipeline.py.
    config.STREAMING_REDECODE_OFFLINE = False
    pipeline_mod.NllbTranslator = StubTranslator  # avoid loading the heavy MT model

    stream = build_stream()
    print(f"Streaming test audio: {stream.size / config.SAMPLE_RATE:.1f}s\n")

    display = RecordingDisplay()
    pipe = pipeline_mod.TranslationPipeline(device=None, display=display, streaming=True)
    pipe._capture = FakeCapture(stream, pipe._audio_queue, pipe.stop_event)

    pipe.start()
    pipe._capture.join()

    deadline = time.time() + 120
    while time.time() < deadline:
        if len(display.pairs) >= 1 and pipe._audio_queue.empty():
            time.sleep(2.0)
            break
        time.sleep(0.5)

    pipe.stop()

    print(
        f"partials={len(display.partials)} finals={len(display.finals)} "
        f"targets={len(display.targets)} pairs={len(display.pairs)}"
    )
    if display.partials:
        print("sample partial:", display.partials[0][:40])
    if display.pairs:
        print("recognized sentences + pairs:")
        for jp, vi in display.pairs:
            print(f"  JP: {jp}")
            print(f"  VI: {vi}")

    ok = (
        len(display.partials) > 0
        and len(display.pairs) >= 1
        and all(jp and vi for jp, vi in display.pairs)
        and all(vi == f"[vi] {jp[:12]}" for jp, vi in display.pairs)
        and any(has_japanese(jp) for jp, _ in display.pairs)
        and not any("[Translate error]" in m for m in display.infos)
    )
    if any("[Translate error]" in m for m in display.infos):
        print("UNEXPECTED translate errors:",
              [m for m in display.infos if "[Translate error]" in m])
    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
