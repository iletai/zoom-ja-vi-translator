"""Japanese speech recognition via ReazonSpeech k2-v2 and sherpa-onnx."""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import sherpa_onnx

import config

logger = logging.getLogger(__name__)


class JapaneseASR:
    """Offline Japanese ASR wrapper for 16 kHz float32 audio."""

    SAMPLE_RATE = 16_000
    FEATURE_DIM = 80
    SILENCE_THRESHOLD = 1e-6

    def __init__(self) -> None:
        model_files = self._find_model_files(Path(config.ASR_MODEL_DIR))

        recognizer_kwargs = dict(
            encoder=str(model_files["encoder"]),
            decoder=str(model_files["decoder"]),
            joiner=str(model_files["joiner"]),
            tokens=str(model_files["tokens"]),
            num_threads=config.ASR_NUM_THREADS,
            provider=config.ASR_PROVIDER,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=self.FEATURE_DIM,
            decoding_method=config.ASR_DECODING_METHOD,
        )

        hotwords_path = Path(config.ASR_HOTWORDS_FILE)
        if hotwords_path.is_file() and config.ASR_DECODING_METHOD == "modified_beam_search":
            recognizer_kwargs["hotwords_file"] = str(hotwords_path)
            recognizer_kwargs["hotwords_score"] = config.ASR_HOTWORDS_SCORE

        # blank_penalty reduces dropped onsets, but only some sherpa-onnx builds
        # accept it — probe the signature so an older wheel doesn't raise.
        blank_penalty = float(getattr(config, "ASR_BLANK_PENALTY", 0.0))
        if blank_penalty > 0.0:
            try:
                import inspect

                params = inspect.signature(
                    sherpa_onnx.OfflineRecognizer.from_transducer
                ).parameters
                if "blank_penalty" in params:
                    recognizer_kwargs["blank_penalty"] = blank_penalty
                else:
                    logger.info("sherpa-onnx build has no blank_penalty param; skipping")
            except (TypeError, ValueError):
                pass

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**recognizer_kwargs)
        self.warmup()

    def transcribe(self, audio_float32: np.ndarray) -> str:
        """Transcribe a 16 kHz mono float32 waveform into Japanese text."""
        audio = np.asarray(audio_float32, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""

        duration_s = audio.size / self.SAMPLE_RATE
        is_silence = bool(np.max(np.abs(audio)) <= self.SILENCE_THRESHOLD)
        # Normalize the whole utterance to a consistent loudness before decoding
        # (helps the recognizer's features; no-op when disabled or on silence).
        if not is_silence:
            from src.audio_enrich import normalize_utterance

            audio = normalize_utterance(audio)
        stream = self.recognizer.create_stream()
        try:
            stream.accept_waveform(self.SAMPLE_RATE, audio)
            self.recognizer.decode_stream(stream)
            text = getattr(stream.result, "text", "").strip()
        finally:
            del stream  # explicit dealloc: sherpa-onnx C++ frees native tensors

        if text and not is_silence:
            logger.debug("Transcribed %.2fs audio -> %s", duration_s, text)
        return "" if is_silence else text

    def warmup(self) -> None:
        """Run one short decode so the first real utterance has lower latency."""
        self.transcribe(np.zeros(int(0.5 * self.SAMPLE_RATE), dtype=np.float32))

    @classmethod
    def _find_model_files(cls, model_dir: Path) -> dict[str, Path]:
        if not model_dir.is_dir():
            raise cls._missing_model_error(model_dir, "model directory")

        return {
            "encoder": cls._find_onnx(model_dir, "encoder"),
            "decoder": cls._find_onnx(model_dir, "decoder"),
            "joiner": cls._find_onnx(model_dir, "joiner"),
            "tokens": cls._find_tokens(model_dir),
        }

    @classmethod
    def _find_onnx(cls, model_dir: Path, component: str) -> Path:
        pattern = os.path.join(str(model_dir), "**", f"*{component}*.onnx")
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if not matches:
            raise cls._missing_model_error(model_dir, f"{component} ONNX file")

        def sort_key(path: Path) -> tuple[int, int, str]:
            name = path.name.lower()
            return (name != f"{component}.onnx", "int8" not in name, name)

        return sorted(matches, key=sort_key)[0]

    @classmethod
    def _find_tokens(cls, model_dir: Path) -> Path:
        pattern = os.path.join(str(model_dir), "**", "tokens.txt")
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if not matches:
            raise cls._missing_model_error(model_dir, "tokens.txt")
        return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]

    @staticmethod
    def _missing_model_error(model_dir: Path, missing: str) -> FileNotFoundError:
        return FileNotFoundError(
            f"Missing {missing} for ReazonSpeech k2-v2 in {model_dir}. "
            "Expected encoder*.onnx, decoder*.onnx, joiner*.onnx, and tokens.txt. "
            "Run scripts/download_models.py from the project root to download ASR models."
        )


if __name__ == "__main__":
    if Path(config.ASR_MODEL_DIR).is_dir():
        asr = JapaneseASR()
        result = asr.transcribe(np.zeros(JapaneseASR.SAMPLE_RATE, dtype=np.float32))
        print(f"Silence transcription: {result!r}")
    else:
        print(
            f"ASR model directory not found: {config.ASR_MODEL_DIR}. "
            "Run scripts/download_models.py from the project root."
        )
