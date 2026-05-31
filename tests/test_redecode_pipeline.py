"""Hybrid offline re-decode pipeline test (the accuracy-critical path).

Proves the data-loss fix: in re-decode mode the text that gets translated comes
from the strong OFFLINE reazonspeech model (via VAD segmentation), NOT the weak
online streaming model. The online model garbles these same wavs (drops the 気象庁
onset -> "J|...", mangles ヤンバルクイナ); the offline model transcribes them
faithfully. We feed the 5 real Japanese wavs exactly like AudioCapture does and
assert the committed Japanese contains the hard tokens the online path loses, with
correct JP/VI pairing and no silent drops.

Runs the REAL online + offline ASR models; only the NLLB translator is stubbed
(keeps the test fast and focused on ASR fidelity + wiring). Slow (~1-2 min).
"""
from __future__ import annotations

import re
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
from src.sentence_aggregator import SentenceAggregator  # noqa: E402

WAV_DIR = ROOT / "test_audio" / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" / "test_wavs"
BLOCK_SECONDS = config.CAPTURE_BLOCK_SECONDS


def load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class StubTranslator:
    """Instant fake translator: echoes a tagged JP prefix so we can verify the
    JP/VI pairing without paying NLLB latency. Mirrors translate_many 1:1."""

    def translate(self, text: str) -> str:
        return f"[vi] {text[:12]}"

    def translate_many(self, texts: list[str]) -> list[str]:
        return [self.translate(t) for t in texts]

    def warmup(self) -> None:
        pass


class RecordingDisplay(SubtitleDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[tuple[int | None, str]] = []
        self.targets: list[tuple[int | None, str, str | None]] = []
        self.infos: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def show_source(self, japanese: str, seq=None) -> None:
        self.sources.append((seq, japanese))

    def show_source_partial(self, committed: str, tail: str = "") -> None:
        pass  # online partials are display-only noise here

    def finalize_source(self, japanese: str) -> None:
        pass

    def show_target(self, vietnamese: str, japanese=None, seq=None) -> None:
        self.targets.append((seq, vietnamese, japanese))

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self.sources.append((None, japanese))
        self.targets.append((None, vietnamese, japanese))


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
            # Pace roughly like real capture so VAD silence timing is realistic.
            self.stop_event.wait(BLOCK_SECONDS * 0.5)
        self.stop_event.wait(0.1)


def build_stream() -> tuple[np.ndarray, int]:
    """All 5 wavs separated by 0.8 s silence so VAD finalizes one utterance each."""
    wavs = sorted(WAV_DIR.glob("*.wav"))
    gap = np.zeros(int(config.SAMPLE_RATE * 0.8), dtype=np.float32)
    parts: list[np.ndarray] = [gap]
    for wav in wavs:
        parts.append(load_wav_float32(wav))
        parts.append(gap)
    return np.concatenate(parts), len(wavs)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def replay_aggregator(raw_fragments: list[str]) -> list[str]:
    agg = SentenceAggregator()
    emitted: list[str] = []
    for fragment in raw_fragments:
        emitted.extend(agg.add(fragment))
    emitted.extend(agg.flush())
    return emitted


def main() -> int:
    config.STREAMING_REDECODE_OFFLINE = True  # force the hybrid path under test
    pipeline_mod.NllbTranslator = StubTranslator

    stream, n_wavs = build_stream()
    print(f"Re-decode test audio: {stream.size / config.SAMPLE_RATE:.1f}s, {n_wavs} wavs\n")

    display = RecordingDisplay()
    pipe = pipeline_mod.TranslationPipeline(device=None, display=display, streaming=True)
    if not getattr(pipe, "_redecode", False):
        print("FAIL: pipeline did not enter re-decode mode")
        return 1

    raw_asr_fragments: list[str] = []
    original_transcribe = pipe.asr.transcribe

    def recording_transcribe(utterance) -> str:
        text = original_transcribe(utterance).strip()
        if text:
            raw_asr_fragments.append(text)
        return text

    pipe.asr.transcribe = recording_transcribe
    pipe._capture = FakeCapture(stream, pipe._audio_queue, pipe.stop_event)

    pipe.start()
    pipe._capture.join()

    # Wait until audio drained AND we have at least one target per wav (or timeout).
    deadline = time.time() + 180
    while time.time() < deadline:
        if pipe._audio_queue.empty() and len(display.targets) >= n_wavs:
            time.sleep(2.0)
            break
        time.sleep(0.5)

    pipe.stop()

    recognized_jp = " ".join(jp for _, jp in display.sources)
    emitted_sources = [jp for _, jp in display.sources]
    expected_sources = replay_aggregator(raw_asr_fragments)
    raw_concat = normalize_text("".join(raw_asr_fragments))
    emitted_concat = normalize_text("".join(emitted_sources))
    replay_concat = normalize_text("".join(expected_sources))
    print(f"raw_asr_fragments={len(raw_asr_fragments)} sources={len(display.sources)} targets={len(display.targets)}")
    for i, raw in enumerate(raw_asr_fragments, 1):
        print(f"  [raw={i}] ASR: {raw}")
    for seq, jp in display.sources:
        print(f"  [seq={seq}] JP: {jp}")

    # --- Assertions -------------------------------------------------------
    checks: list[tuple[str, bool]] = []

    # 1. Offline fidelity: hard tokens the ONLINE model loses must be present.
    checks.append(("contains 気象庁 (online drops this onset)", "気象庁" in recognized_jp))
    checks.append(("contains ヤンバルクイナ (online garbles this)", "ヤンバルクイナ" in recognized_jp))

    # 2. No online-style onset-drop garbage marker.
    checks.append(("no 'J|' garbage artifact", "J|" not in recognized_jp))

    # 3. Got an utterance for (nearly) every wav — nothing wholesale dropped.
    checks.append((f"recognized >= {n_wavs - 1} utterances", len(display.sources) >= n_wavs - 1))

    # 3b. Hard offline aggregation contract: emitted source sentences must be a
    # lossless re-segmentation of the raw ASR fragments (no dropped or duplicated
    # characters, ignoring display whitespace only).
    checks.append(("raw ASR fragments were observed", len(raw_asr_fragments) >= n_wavs - 1))
    checks.append(("aggregated source concat == raw ASR concat", emitted_concat == raw_concat))
    checks.append(("SentenceAggregator replay itself is lossless", replay_concat == raw_concat))
    checks.append(("sensible aggregated sentence count",
                   2 <= len(emitted_sources) <= max(n_wavs * 2, len(raw_asr_fragments) + 2)))
    checks.append(("aggregation merged raw fragments", len(emitted_sources) < len(raw_asr_fragments)))
    checks.append(("known 気象庁 fragments merged",
                   any("気象庁は" in s and "雪や路面" in s for s in emitted_sources)))
    checks.append(("known H2A fragments merged",
                   any("打ち上げの成功率" in s and "一回の打ち上げ費用" in s for s in emitted_sources)))

    # 4. Every committed source is non-empty Japanese.
    def has_japanese(t: str) -> bool:
        return any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in t)
    checks.append(("all sources non-empty JP", all(jp and has_japanese(jp) for _, jp in display.sources)))

    # 5. JP/VI pairing: every target seq maps back to a committed source seq.
    source_seqs = {seq for seq, _ in display.sources if seq is not None}
    target_seqs = [seq for seq, _, _ in display.targets if seq is not None]
    checks.append(("every target seq has a source seq",
                   all(s in source_seqs for s in target_seqs) and len(target_seqs) > 0))

    # 6. No translate errors surfaced to the display.
    checks.append(("no [Translate error]",
                   not any("[Translate error]" in m for m in display.infos)))

    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    print(f"\n=== RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
