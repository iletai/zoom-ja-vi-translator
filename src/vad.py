"""Voice activity based utterance segmentation for real-time audio streams."""
from __future__ import annotations

from collections import deque
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
        self._min_silence_at_max_frames = max(1, 98 // self.frame_ms)
        self._energy_gate = bool(getattr(config, "VAD_ENERGY_GATE", False))
        self._noise_floor: Optional[float] = None
        self._energy_margin = float(getattr(config, "VAD_ENERGY_MARGIN_RMS", 120.0)) / 32768.0
        self._energy_multiplier = float(getattr(config, "VAD_ENERGY_MULTIPLIER", 1.8))
        self._energy_alpha = float(getattr(config, "VAD_ENERGY_NOISE_ALPHA", 0.10))

        # Pre-roll "collar": keep this many ms of audio leading up to a detected
        # speech onset and prepend it to the utterance. Aggressive VAD settings
        # (needed to find boundaries in continuous/background audio) classify the
        # quiet leading mora of a word as non-speech and would clip it (verified:
        # 気象庁 -> 町長); the collar restores that onset so the offline ASR sees
        # the whole word.
        preroll_ms = int(getattr(config, "VAD_PREROLL_MS", 0))
        preroll_frames = max(0, preroll_ms // self.frame_ms)

        self._vad = webrtcvad.Vad(int(config.VAD_AGGRESSIVENESS))
        self._leftover = np.empty(0, dtype=np.float32)
        self._utterance_frames: list[np.ndarray] = []
        self._utterance_voiced: list[bool] = []
        self._utterance_rms: list[float] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
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
            raw_voiced = self._vad.is_speech(pcm_bytes, self.sample_rate)
            rms = self._frame_rms(frame)
            is_voiced = self._effective_voiced(raw_voiced, rms)

            if is_voiced:
                if not self._utterance_frames and self._preroll:
                    # Seed the onset with the collar so the leading mora survives.
                    for pf in self._preroll:
                        self._start_or_append(pf, voiced=False, rms=self._frame_rms(pf))
                    self._preroll.clear()
                self._start_or_append(frame, voiced=True, rms=rms)
                self._current_silence_ms = 0
                self._current_voiced_ms += self.frame_ms
            elif self._utterance_frames:
                # Keep short internal/trailing pauses so ASR gets natural context.
                self._start_or_append(frame, voiced=False, rms=rms)
                self._current_silence_ms += self.frame_ms
            else:
                # Idle (no active utterance): keep a rolling pre-onset collar.
                self._preroll.append(frame)

            if not self._utterance_frames:
                continue

            if self._current_silence_ms >= self.silence_ms:
                utterance = self._finish_current()
            elif self._current_utterance_ms >= self.max_utterance_ms:
                utterance = self._min_cut()
            else:
                utterance = None
            if utterance is not None:
                completed.append(utterance)

        return completed

    def flush(self) -> Optional[np.ndarray]:
        """Return any in-progress utterance, or None if nothing is buffered."""
        if self._leftover.size and self._utterance_frames:
            # On shutdown, preserve trailing samples that were too short for VAD.
            self._utterance_frames.append(self._leftover.copy())
            self._utterance_voiced.append(False)
            self._utterance_rms.append(self._frame_rms(self._leftover))
        self._leftover = np.empty(0, dtype=np.float32)
        return self._finish_current()

    def _effective_voiced(self, raw_voiced: bool, rms: float) -> bool:
        if not self._energy_gate:
            return raw_voiced

        if not raw_voiced:
            if self._noise_floor is None:
                self._noise_floor = rms
            else:
                self._noise_floor = ((1.0 - self._energy_alpha) * self._noise_floor) + (
                    self._energy_alpha * rms
                )

        nf = self._noise_floor if self._noise_floor is not None else 0.0
        # RMS stays in float32 audio scale (0..1); the static margin was converted
        # from PCM-16 units during initialization.
        threshold = max(self._energy_margin, nf * self._energy_multiplier, nf + self._energy_margin)
        return raw_voiced and rms >= threshold

    def _start_or_append(self, frame: np.ndarray, voiced: bool, rms: float) -> None:
        self._utterance_frames.append(frame)
        self._utterance_voiced.append(bool(voiced))
        self._utterance_rms.append(float(rms))
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

    def _min_cut(self) -> Optional[np.ndarray]:
        if not self._utterance_frames:
            return None

        n = len(self._utterance_frames)
        half = n // 2
        cut_idx: Optional[int] = None
        carry_start: Optional[int] = None

        best_start = -1
        best_end = -1
        i = half
        while i < n:
            if self._utterance_voiced[i]:
                i += 1
                continue
            start = i
            while i < n and not self._utterance_voiced[i]:
                i += 1
            end = i
            run_len = end - start
            best_len = best_end - best_start
            if run_len >= self._min_silence_at_max_frames and (run_len > best_len or run_len == best_len):
                best_start, best_end = start, end

        if best_start >= 0:
            cut_idx = max(1, best_start)
            carry_start = best_end
        else:
            rms_tail = self._utterance_rms[half:]
            min_offset = int(np.argmin(rms_tail)) if rms_tail else 0
            cut_idx = max(1, half + min_offset)
            carry_start = cut_idx + 1

        emitted_frames = self._utterance_frames[:cut_idx]
        carry_frames = self._utterance_frames[carry_start:]
        carry_voiced = self._utterance_voiced[carry_start:]
        carry_rms = self._utterance_rms[carry_start:]
        if len(carry_frames) >= n:
            carry_frames = []
            carry_voiced = []
            carry_rms = []
        assert len(carry_frames) < n

        emitted = np.concatenate(emitted_frames).astype(np.float32, copy=False)
        self._reset_utterance()
        self._reseed_carry(carry_frames, carry_voiced, carry_rms)
        return emitted

    def _reseed_carry(
        self,
        frames: list[np.ndarray],
        voiced_flags: list[bool],
        rms_values: list[float],
    ) -> None:
        self._utterance_frames = list(frames)
        self._utterance_voiced = list(voiced_flags)
        self._utterance_rms = list(rms_values)
        self._current_utterance_ms = len(frames) * self.frame_ms
        self._current_voiced_ms = sum(1 for voiced in voiced_flags if voiced) * self.frame_ms
        trailing_silence = 0
        for voiced in reversed(voiced_flags):
            if voiced:
                break
            trailing_silence += 1
        self._current_silence_ms = trailing_silence * self.frame_ms
        self._preroll.clear()

    @staticmethod
    def _frame_rms(frame: np.ndarray) -> float:
        arr = np.asarray(frame, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr ** 2)))

    def _reset_utterance(self) -> None:
        self._utterance_frames = []
        self._utterance_voiced = []
        self._utterance_rms = []
        self._preroll.clear()
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
