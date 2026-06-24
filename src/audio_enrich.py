"""Light, dependency-free audio conditioning applied before VAD/ASR.

Real meeting audio is messy: quiet speakers, HVAC/fan rumble, mic bumps, and
wildly different levels between participants. Feeding that straight to the
recognizer costs accuracy. This stage does two cheap, robust things per captured
block, using only numpy:

1. **High-pass filter** — a one-pole DC-blocker that removes sub-speech-band
   rumble and any DC offset (which otherwise biases the VAD energy gate).
2. **Soft AGC** — gains quiet blocks up toward a target RMS so the ASR sees a
   consistent loudness, with a hard gain cap and a noise floor so silence/room
   tone is never amplified into hiss.

Both are conservative and reversible via config (``AUDIO_*`` knobs / ``ZT_AUDIO_*``
env). The filter is stateful across blocks (it must be, to avoid a discontinuity
at every 200 ms boundary), so one ``AudioEnricher`` instance belongs to one
capture stream and is not thread-safe.

It deliberately does NOT do spectral denoise / noise suppression: that needs a
heavier dependency (e.g. rnnoise/noisereduce) and can smear consonants, hurting
Japanese ASR more than the broadband noise it removes. High-pass + AGC are the
high-value, low-risk wins; richer denoise can be added later behind its own flag.
"""
from __future__ import annotations

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)


class AudioEnricher:
    """Per-stream audio conditioner: high-pass + soft AGC. Not thread-safe."""

    def __init__(
        self,
        sample_rate: int | None = None,
        *,
        highpass_hz: float | None = None,
        target_rms: float | None = None,
        max_gain: float | None = None,
        noise_floor_rms: float | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate or config.SAMPLE_RATE)
        self.highpass_hz = float(
            highpass_hz if highpass_hz is not None else getattr(config, "AUDIO_HIGHPASS_HZ", 80.0)
        )
        self.target_rms = float(
            target_rms if target_rms is not None else getattr(config, "AUDIO_TARGET_RMS", 0.05)
        )
        self.max_gain = float(
            max_gain if max_gain is not None else getattr(config, "AUDIO_MAX_GAIN", 8.0)
        )
        self.noise_floor_rms = float(
            noise_floor_rms
            if noise_floor_rms is not None
            else getattr(config, "AUDIO_NOISE_FLOOR_RMS", 0.005)
        )

        # One-pole high-pass coefficient. R close to 1 → lower cutoff. Derived
        # from the standard DC-blocker: y[n] = x[n] - x[n-1] + R*y[n-1].
        if self.highpass_hz > 0:
            self._hp_r = float(np.exp(-2.0 * np.pi * self.highpass_hz / self.sample_rate))
        else:
            self._hp_r = 0.0
        # Filter memory carried across blocks (prev input + prev output sample).
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0

        # Prefer scipy's C-level IIR (lfilter) when available — the recursion is
        # not numpy-vectorizable, so the pure-Python fallback loop is much slower
        # (still fine at 16 kHz/200 ms blocks, but avoid it when we can).
        try:
            from scipy.signal import lfilter  # noqa: F401

            self._lfilter = lfilter
            # Transfer function of y[n] = x[n] - x[n-1] + R*y[n-1]:
            #   b = [1, -1], a = [1, -R]
            self._hp_b = np.array([1.0, -1.0], dtype=np.float64)
            self._hp_a = np.array([1.0, -self._hp_r], dtype=np.float64)
        except Exception:  # noqa: BLE001 - scipy optional
            self._lfilter = None

    def process(self, block: np.ndarray) -> np.ndarray:
        """Condition one float32 mono block; returns a new float32 array.

        Safe on empty input. Always returns audio clipped to [-1, 1] so the
        downstream PCM16 conversion never wraps.
        """
        audio = np.asarray(block, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio

        if self._hp_r > 0.0:
            audio = self._highpass(audio)

        if self.target_rms > 0.0:
            audio = self._agc(audio)

        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        """Stateful one-pole DC-blocking high-pass: y[n]=x[n]-x[n-1]+R*y[n-1].

        Uses scipy.lfilter (C speed) with carried filter state when available,
        else a correct pure-Python fallback. State persists across blocks so
        there is no discontinuity at the 200 ms block boundaries.
        """
        if self._lfilter is not None:
            # zi holds [R*y[-1] - x[-1]] for this b/a; seed from carried samples.
            zi = np.array([self._hp_r * self._hp_prev_y - self._hp_prev_x], dtype=np.float64)
            y, zf = self._lfilter(self._hp_b, self._hp_a, x.astype(np.float64), zi=zi)
            self._hp_prev_x = float(x[-1])
            self._hp_prev_y = float(y[-1])
            return y.astype(np.float32)

        r = self._hp_r
        n = x.size
        y = np.empty(n, dtype=np.float32)
        prev_x = self._hp_prev_x
        prev_y = self._hp_prev_y
        for i in range(n):
            xi = float(x[i])
            yi = xi - prev_x + r * prev_y
            y[i] = yi
            prev_x = xi
            prev_y = yi
        self._hp_prev_x = prev_x
        self._hp_prev_y = prev_y
        return y

    def _agc(self, x: np.ndarray) -> np.ndarray:
        """Scale toward target RMS, capped, skipping silence."""
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms < self.noise_floor_rms:
            return x  # silence / room tone — don't amplify
        gain = self.target_rms / rms
        if gain > self.max_gain:
            gain = self.max_gain
        elif gain < 1.0:
            # Only gain up; loud blocks are left alone (clip guard handles peaks).
            return x
        return (x * np.float32(gain)).astype(np.float32)


def make_enricher(sample_rate: int | None = None) -> "AudioEnricher | None":
    """Return an enricher if AUDIO_ENRICH is enabled, else None (no-op caller)."""
    if not getattr(config, "AUDIO_ENRICH", False):
        return None
    enr = AudioEnricher(sample_rate)
    logger.info(
        "Audio enrichment on: highpass=%.0fHz target_rms=%.3f max_gain=%.1f",
        enr.highpass_hz, enr.target_rms, enr.max_gain,
    )
    return enr
