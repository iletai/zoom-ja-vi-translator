"""Component tests on real Japanese speech audio.

Exercises ASR, VAD segmentation and NLLB translation against the sherpa-onnx
ReazonSpeech sample wavs (real Japanese utterances) so we verify the actual
models end-to-end, not just syntax.
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.asr import JapaneseASR  # noqa: E402
from src.translator import NllbTranslator  # noqa: E402
from src.vad import VadSegmenter  # noqa: E402

WAV_DIR = ROOT / "test_audio" / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" / "test_wavs"


def load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == config.SAMPLE_RATE, w.getframerate()
        assert w.getnchannels() == 1
        frames = w.readframes(w.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return pcm


def main() -> int:
    wavs = sorted(WAV_DIR.glob("*.wav"))
    assert wavs, f"no wavs in {WAV_DIR}"
    print(f"Found {len(wavs)} Japanese test wavs\n")

    print("=== Loading models ===")
    t0 = time.time()
    asr = JapaneseASR()
    translator = NllbTranslator()
    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    failures = 0
    for wav in wavs:
        audio = load_wav_float32(wav)

        t0 = time.time()
        japanese = asr.transcribe(audio).strip()
        asr_ms = (time.time() - t0) * 1000

        t0 = time.time()
        vietnamese = translator.translate(japanese).strip() if japanese else ""
        mt_ms = (time.time() - t0) * 1000

        # VAD must find at least one utterance in real speech.
        seg = VadSegmenter()
        utterances = []
        block = int(config.SAMPLE_RATE * 0.2)
        for off in range(0, audio.size, block):
            utterances.extend(seg.push(audio[off : off + block]))
        tail = seg.flush()
        if tail is not None:
            utterances.append(tail)

        ok = bool(japanese) and bool(vietnamese) and len(utterances) >= 1
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {wav.name}  (dur={audio.size / config.SAMPLE_RATE:.1f}s, "
              f"asr={asr_ms:.0f}ms, mt={mt_ms:.0f}ms, vad_utts={len(utterances)})")
        print(f"   JA: {japanese}")
        print(f"   VI: {vietnamese}\n")

    print("=== Translation-only sanity (canned JA meeting phrases) ===")
    phrases = ["本日の会議を始めます。", "資料を共有します。", "質問はありますか？"]
    for jp in phrases:
        vi = translator.translate(jp).strip()
        ok = bool(vi)
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {jp} -> {vi}")

    print(f"\n=== RESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
