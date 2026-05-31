"""Streaming (online) Japanese ASR via sherpa-onnx.

Unlike the offline :class:`~src.asr.JapaneseASR`, which can only transcribe a
finished utterance, the online recognizer consumes audio incrementally and
exposes a growing *partial* hypothesis after every chunk. This lets the UI show
recognized Japanese within a few hundred milliseconds of speech — the key to a
low-latency, "live caption" feel — while built-in endpoint detection marks where
one utterance ends so the translator can run on a complete sentence.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import sherpa_onnx

try:
    import config
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config


class StreamingJapaneseASR:
    """Incremental Japanese ASR over a single continuous 16 kHz audio stream."""

    SAMPLE_RATE = 16_000
    FEATURE_DIM = 80

    def __init__(self) -> None:
        files = self._find_model_files(Path(config.STREAMING_ASR_MODEL_DIR))
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(files["tokens"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            num_threads=config.STREAMING_ASR_NUM_THREADS,
            provider=config.ASR_PROVIDER,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=self.FEATURE_DIM,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=config.STREAMING_RULE1_SILENCE,
            rule2_min_trailing_silence=config.STREAMING_RULE2_SILENCE,
            rule3_min_utterance_length=config.STREAMING_RULE3_UTTERANCE,
        )
        self.stream = self.recognizer.create_stream()
        self._overlap = np.zeros(0, dtype=np.float32)

    def accept(self, samples: np.ndarray) -> None:
        """Feed a block of 16 kHz mono float32 audio and decode what is ready."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        self.stream.accept_waveform(self.SAMPLE_RATE, audio)
        self._append_overlap(audio)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def partial(self) -> str:
        """Return the current (possibly incomplete) Japanese hypothesis."""
        return self.recognizer.get_result(self.stream).strip()

    def at_endpoint(self) -> bool:
        """True when the recognizer has detected the end of an utterance."""
        return self.recognizer.is_endpoint(self.stream)

    def reset(self) -> None:
        """Start a fresh utterance after the previous one was finalized."""
        self.recognizer.reset(self.stream)
        if self._overlap_window_samples() <= 0:
            self._overlap = np.zeros(0, dtype=np.float32)
            return

        overlap = self._overlap.copy()
        self._overlap = np.zeros(0, dtype=np.float32)
        if overlap.size == 0:
            return

        self.stream.accept_waveform(self.SAMPLE_RATE, overlap)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def finalize_tail(self) -> None:
        """Flush trailing audio so the last partial reflects all fed samples."""
        tail = np.zeros(int(0.5 * self.SAMPLE_RATE), dtype=np.float32)
        self.stream.accept_waveform(self.SAMPLE_RATE, tail)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def _append_overlap(self, audio: np.ndarray) -> None:
        max_samples = self._overlap_window_samples()
        if max_samples <= 0:
            self._overlap = np.zeros(0, dtype=np.float32)
            return
        if audio.size >= max_samples:
            self._overlap = audio[-max_samples:].copy()
            return
        self._overlap = np.concatenate((self._overlap, audio))[-max_samples:]

    def _overlap_window_samples(self) -> int:
        return int(max(0.0, config.STREAMING_AUDIO_OVERLAP_SEC) * self.SAMPLE_RATE)

    @classmethod
    def _find_model_files(cls, model_dir: Path) -> dict[str, Path]:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Streaming ASR model directory not found: {model_dir}. "
                "Run scripts/download_models.py to download the streaming model."
            )
        return {
            "encoder": cls._find_onnx(model_dir, "encoder"),
            "decoder": cls._find_onnx(model_dir, "decoder"),
            "joiner": cls._find_onnx(model_dir, "joiner"),
            "tokens": cls._find_tokens(model_dir),
        }

    @classmethod
    def _find_onnx(cls, model_dir: Path, component: str) -> Path:
        pattern = os.path.join(str(model_dir), "**", f"*{component}*.onnx")
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if not matches:
            raise FileNotFoundError(
                f"Missing {component} ONNX file for the streaming model in {model_dir}."
            )
        # Prefer the int8 build (smaller / faster) when both are present.
        return sorted(matches, key=lambda p: ("int8" not in p.name.lower(), p.name))[0]

    @classmethod
    def _find_tokens(cls, model_dir: Path) -> Path:
        pattern = os.path.join(str(model_dir), "**", "tokens.txt")
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if not matches:
            raise FileNotFoundError(f"Missing tokens.txt for the streaming model in {model_dir}.")
        return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]


if __name__ == "__main__":
    if Path(config.STREAMING_ASR_MODEL_DIR).is_dir():
        asr = StreamingJapaneseASR()
        asr.accept(np.zeros(StreamingJapaneseASR.SAMPLE_RATE, dtype=np.float32))
        print(f"Streaming partial on silence: {asr.partial()!r}")
    else:
        print(f"Streaming model not found: {config.STREAMING_ASR_MODEL_DIR}")
