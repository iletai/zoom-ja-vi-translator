"""Cloud speech-translation backend (Azure Speech Translation).

This is an OPTIONAL low-latency backend. Unlike the local CPU pipeline
(sherpa-onnx ASR + NLLB CTranslate2 MT), which floors at several seconds per
sentence, Azure Speech Translation performs streaming Japanese recognition AND
Japanese -> Vietnamese translation in a single GPU-backed service call, with
sub-second interim results.

Why Azure: it is the only major provider that does JA -> VI in one streaming
SDK call, has a generous free tier (5 audio hours / month), and the cleanest
Python SDK. (DeepL has no Vietnamese; Zoom's native translated captions have no
Vietnamese target either.)

Privacy trade-off: audio is sent to Microsoft Azure. For 100% offline use, stay
on the local backend.

The event-handling logic (``_handle_partial`` / ``_handle_final``) and the
float32 -> 16-bit PCM conversion are deliberately separated from the Azure SDK
wiring so they can be unit-tested without a network connection or API key.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np

import config
from src.display import SubtitleDisplay


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert mono float32 samples in [-1, 1] to little-endian 16-bit PCM bytes.

    Azure's ``PushAudioInputStream`` expects raw 16-bit signed PCM. Samples are
    clipped before scaling so out-of-range values can't wrap around to the
    opposite polarity.
    """
    if samples is None:
        return b""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:  # collapse any stray channel axis to mono
        arr = arr.reshape(-1)
    if arr.size == 0:
        return b""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class AzureSpeechTranslator:
    """Stream system audio to Azure and render live Vietnamese subtitles.

    Usage::

        translator = AzureSpeechTranslator(display)
        translator.start()
        translator.push(float32_block)   # repeatedly, from the capture thread
        ...
        translator.stop()
    """

    def __init__(
        self,
        display: SubtitleDisplay,
        *,
        key: str | None = None,
        region: str | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        sample_rate: int | None = None,
        on_fatal: Callable[[str], None] | None = None,
    ) -> None:
        self.display = display
        self.key = key or config.AZURE_SPEECH_KEY
        self.region = region or config.AZURE_SPEECH_REGION
        self.source_lang = source_lang or config.CLOUD_SOURCE_LANG
        self.target_lang = target_lang or config.CLOUD_TARGET_LANG
        self.sample_rate = int(sample_rate or config.SAMPLE_RATE)
        self.on_fatal = on_fatal

        self._last_partial = ""
        self._push_stream = None
        self._recognizer = None
        self._started = False
        # Guards display state and stream access against the Azure SDK callback
        # threads (recognizing/recognized/canceled) and the capture worker.
        self._lock = threading.Lock()
        self._stopping = False

        if not self.key or not self.region:
            raise ValueError(
                "Azure cloud backend requires credentials. Set AZURE_SPEECH_KEY "
                "and AZURE_SPEECH_REGION environment variables (free tier: create "
                "a Speech resource at https://portal.azure.com — F0 = 5 hours/month "
                "free)."
            )

    # ─── SDK wiring ──────────────────────────────────────────────────────
    def _build_recognizer(self):
        # Imported lazily so the optional dependency is only required in cloud
        # mode (``pip install -r requirements-cloud.txt``).
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "azure-cognitiveservices-speech is not installed. Install the "
                "cloud extras: pip install -r requirements-cloud.txt"
            ) from exc

        translation_config = speechsdk.translation.SpeechTranslationConfig(
            subscription=self.key, region=self.region
        )
        translation_config.speech_recognition_language = self.source_lang
        translation_config.add_target_language(self.target_lang)

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate, bits_per_sample=16, channels=1
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config, audio_config=audio_config
        )
        recognizer.recognizing.connect(self._on_recognizing)
        recognizer.recognized.connect(self._on_recognized)
        recognizer.canceled.connect(self._on_canceled)
        return recognizer

    def _on_recognizing(self, evt) -> None:
        result = evt.result
        self._handle_partial(getattr(result, "text", "") or "")

    def _on_recognized(self, evt) -> None:
        result = evt.result
        japanese = getattr(result, "text", "") or ""
        translations = getattr(result, "translations", None) or {}
        vietnamese = translations.get(self.target_lang, "") if translations else ""
        self._handle_final(japanese, vietnamese)

    def _on_canceled(self, evt) -> None:
        # A cancellation during intentional shutdown is expected; ignore it.
        with self._lock:
            if self._stopping:
                return
        detail = getattr(evt, "error_details", None) or getattr(evt, "reason", "")
        message = f"[Azure cancelled] {detail}"
        self.display.info(message)
        # Non-user cancellation (bad key, quota, dropped connection) is fatal:
        # signal the pipeline to stop instead of silently capturing forever.
        if self.on_fatal is not None:
            self.on_fatal(message)

    # ─── Display logic (pure, unit-testable) ─────────────────────────────
    def _handle_partial(self, japanese: str) -> None:
        japanese = japanese.strip()
        with self._lock:
            if self._stopping or not japanese or japanese == self._last_partial:
                return
            self._last_partial = japanese
        self.display.show_source_partial(japanese)

    def _handle_final(self, japanese: str, vietnamese: str) -> None:
        japanese = japanese.strip()
        vietnamese = vietnamese.strip()
        with self._lock:
            if self._stopping or not japanese:
                return
            self._last_partial = ""
        self.display.finalize_source(japanese)
        self.display.show_target(vietnamese or "(...)")

    # ─── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        try:
            self._recognizer = self._build_recognizer()
            self._recognizer.start_continuous_recognition_async().get()
            self._started = True
        except Exception:
            # Clean up partially built native objects so we don't leak them.
            self.stop()
            raise

    def push(self, samples: np.ndarray) -> None:
        """Feed one captured mono float32 block to the recognizer."""
        with self._lock:
            if self._stopping or self._push_stream is None:
                return
            stream = self._push_stream
        pcm = float32_to_pcm16(samples)
        if pcm:
            stream.write(pcm)

    def stop(self) -> None:
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            stream = self._push_stream
            self._push_stream = None
        try:
            if stream is not None:
                stream.close()
            if self._recognizer is not None:
                self._recognizer.stop_continuous_recognition_async().get()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass
        finally:
            self._started = False
