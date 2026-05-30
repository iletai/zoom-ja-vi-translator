from __future__ import annotations

import queue
import threading

import numpy as np
import soundcard as sc

import config


_LOOPBACK_NAME_HINTS = ("blackhole", "loopback", "monitor")


def _device_name(device: object) -> str:
    return str(getattr(device, "name", device))


def _is_loopback(device: object) -> bool:
    return bool(getattr(device, "isloopback", getattr(device, "_isloopback", False)))


def _native_sample_rate(device: object) -> int:
    for attr_name in ("samplerate", "default_samplerate", "_samplerate", "_default_samplerate"):
        value = getattr(device, attr_name, None)
        if callable(value):
            value = value()
        if value:
            return int(round(float(value)))
    return int(config.SAMPLE_RATE)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        if samples.shape[1] == 1:
            return samples[:, 0]
        return samples.mean(axis=1, dtype=np.float32)
    return samples.reshape(-1).astype(np.float32, copy=False)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    target_length = max(1, int(round(samples.size * target_rate / source_rate)))
    if samples.size == 1:
        return np.full(target_length, samples[0], dtype=np.float32)

    source_positions = np.linspace(0.0, samples.size - 1, num=samples.size, dtype=np.float64)
    target_positions = np.linspace(0.0, samples.size - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


class AudioCapture(threading.Thread):
    """Capture loopback audio blocks and enqueue mono 16 kHz float32 samples."""

    def __init__(self, device: object, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.device = device
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            source_rate = _native_sample_rate(self.device)
            block_frames = max(1, int(round(source_rate * config.CAPTURE_BLOCK_SECONDS)))

            with self.device.recorder(samplerate=source_rate, blocksize=block_frames) as recorder:
                while not self.stop_event.is_set():
                    audio = recorder.record(numframes=block_frames)
                    mono = _to_mono_float32(audio)
                    resampled = _resample_linear(mono, source_rate, int(config.SAMPLE_RATE))
                    self._put_drop_oldest(resampled)
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def _put_drop_oldest(self, audio: np.ndarray) -> None:
        audio = audio.astype(np.float32, copy=False)
        # Dropping stale blocks keeps translation latency bounded under load.
        while not self.stop_event.is_set():
            self._trim_to_configured_queue_size()
            try:
                self.output_queue.put_nowait(audio)
                return
            except queue.Full:
                self._drop_oldest()

    def _trim_to_configured_queue_size(self) -> None:
        while self.output_queue.qsize() >= int(config.CAPTURE_QUEUE_MAXSIZE):
            if not self._drop_oldest():
                return

    def _drop_oldest(self) -> bool:
        try:
            self.output_queue.get_nowait()
            return True
        except queue.Empty:
            return False


def list_devices() -> list[dict[str, object]]:
    """Print and return available input and loopback capture devices."""
    devices = sc.all_microphones(include_loopback=True)
    result: list[dict[str, object]] = []

    for index, device in enumerate(devices):
        is_loopback = _is_loopback(device)
        name = _device_name(device)
        print(f"[{index}] {name} (loopback={is_loopback})")
        result.append(
            {
                "index": index,
                "name": name,
                "is_loopback": is_loopback,
                "device": device,
            }
        )

    return result


def find_loopback_device() -> object | None:
    """Return the best available loopback-style capture device, if any."""
    devices = sc.all_microphones(include_loopback=True)

    for hint in _LOOPBACK_NAME_HINTS:
        for device in devices:
            if hint in _device_name(device).lower():
                return device

    for device in devices:
        if _is_loopback(device):
            return device

    return None


def get_default_microphone() -> object:
    return sc.default_microphone()


if __name__ == "__main__":
    list_devices()
