"""Voice activity based utterance segmentation for real-time audio streams."""
from __future__ import annotations

from typing import Optional

import numpy as np

import config

try:
    import webrtcvad
except ImportError:  # pragma: no cover - allows syntax checks without optional wheel.
    webrtcvad = None


class VadSegmenter:
    """Segment 16 kHz mono float32 audio into utterances using WebRTC VAD."""

    def __init__(self) -> None:
        if webrtcvad is None:
            raise ImportError("webrtcvad is required to use VadSegmenter")

        self.sample_rate = int(config.SAMPLE_RATE)
        self.frame_ms = int(config.VAD_FRAME_MS)
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("config.VAD_FRAME_MS must be one of 10, 20, or 30")

        self.silence_ms = int(config.VAD_SILENCE_MS)
        self.min_utterance_ms = int(config.VAD_MIN_UTTERANCE_MS)
        self.max_utterance_ms = int(config.VAD_MAX_UTTERANCE_MS)
        self.frame_samples = self.sample_rate * self.frame_ms // 1000

        self._vad = webrtcvad.Vad(int(config.VAD_AGGRESSIVENESS))
        self._leftover = np.empty(0, dtype=np.float32)
        self._utterance_frames: list[np.ndarray] = []
        self._current_silence_ms = 0
        self._current_utterance_ms = 0
        self._current_voiced_ms = 0

    def float32_to_pcm16_frames(self, audio_float32: np.ndarray) -> list[tuple[np.ndarray, bytes]]:
        """Convert float32 audio to exact-size PCM16 frames for WebRTC VAD.

        WebRTC VAD only accepts 16-bit mono PCM buffers whose duration is
        exactly 10, 20, or 30 ms, so incomplete trailing samples are retained
        and prepended to the next call.
        """
        audio = np.asarray(audio_float32, dtype=np.float32).reshape(-1)
        if self._leftover.size:
            audio = np.concatenate((self._leftover, audio))

        full_sample_count = (audio.size // self.frame_samples) * self.frame_samples
        if full_sample_count == 0:
            self._leftover = audio.copy()
            return []

        complete = audio[:full_sample_count]
        self._leftover = audio[full_sample_count:].copy()

        frames: list[tuple[np.ndarray, bytes]] = []
        for start in range(0, full_sample_count, self.frame_samples):
            frame = complete[start : start + self.frame_samples].copy()
            # Clip to the expected float range before scaling to signed PCM16.
            pcm16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2")
            frames.append((frame, pcm16.tobytes()))
        return frames

    def push(self, audio_float32: np.ndarray) -> list[np.ndarray]:
        """Feed audio and return all utterances completed by this block."""
        completed: list[np.ndarray] = []

        for frame, pcm_bytes in self.float32_to_pcm16_frames(audio_float32):
            is_voiced = self._vad.is_speech(pcm_bytes, self.sample_rate)

            if is_voiced:
                self._start_or_append(frame)
                self._current_silence_ms = 0
                self._current_voiced_ms += self.frame_ms
            elif self._utterance_frames:
                # Keep short internal/trailing pauses so ASR gets natural context.
                self._start_or_append(frame)
                self._current_silence_ms += self.frame_ms

            if not self._utterance_frames:
                continue

            if (
                self._current_silence_ms >= self.silence_ms
                or self._current_utterance_ms >= self.max_utterance_ms
            ):
                utterance = self._finish_current()
                if utterance is not None:
                    completed.append(utterance)

        return completed

    def flush(self) -> Optional[np.ndarray]:
        """Return any in-progress utterance, or None if nothing is buffered."""
        if self._leftover.size and self._utterance_frames:
            # On shutdown, preserve trailing samples that were too short for VAD.
            self._utterance_frames.append(self._leftover.copy())
        self._leftover = np.empty(0, dtype=np.float32)
        return self._finish_current()

    def _start_or_append(self, frame: np.ndarray) -> None:
        self._utterance_frames.append(frame)
        self._current_utterance_ms += self.frame_ms

    def _finish_current(self) -> Optional[np.ndarray]:
        if not self._utterance_frames:
            return None

        utterance = np.concatenate(self._utterance_frames).astype(np.float32, copy=False)
        voiced_ms = self._current_voiced_ms
        self._reset_utterance()

        # Silence used to detect the endpoint does not count toward the minimum;
        # this prevents very short noise blips plus trailing silence from passing.
        if voiced_ms < self.min_utterance_ms:
            return None
        return utterance

    def _reset_utterance(self) -> None:
        self._utterance_frames = []
        self._current_silence_ms = 0
        self._current_utterance_ms = 0
        self._current_voiced_ms = 0


if __name__ == "__main__":
    if webrtcvad is None:
        print("webrtcvad is not installed; skipping VadSegmenter self-test")
    else:
        sample_rate = int(config.SAMPLE_RATE)
        tone_seconds = 1.0
        silence_seconds = 1.0
        t = np.arange(int(sample_rate * tone_seconds), dtype=np.float32) / sample_rate
        sine = 0.4 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        silence = np.zeros(int(sample_rate * silence_seconds), dtype=np.float32)
        audio = np.concatenate((sine, silence))

        segmenter = VadSegmenter()
        utterances = []
        block_samples = int(sample_rate * 0.2)
        for offset in range(0, audio.size, block_samples):
            utterances.extend(segmenter.push(audio[offset : offset + block_samples]))
        flushed = segmenter.flush()
        if flushed is not None:
            utterances.append(flushed)

        print(f"Detected {len(utterances)} utterance(s)")
