"""Multi-threaded orchestration: capture -> VAD -> ASR -> translate -> display.

Three decoupled stages connected by bounded queues keep the UI responsive and
bound end-to-end latency:

    AudioCapture thread  --audio blocks-->  [audio_queue]
    ASR worker thread    pulls blocks, runs VAD, transcribes complete
                         utterances, then pushes Japanese text -->  [text_queue]
    Translate worker     pulls Japanese text, translates to Vietnamese,
                         and renders the subtitle pair.
"""
from __future__ import annotations

import queue
import threading
from time import monotonic

import config
from src.asr import JapaneseASR
from src.audio_capture import AudioCapture
from src.display import SubtitleDisplay
from src.local_agreement import LocalAgreementBuffer
from src.sentence_aggregator import SentenceAggregator
from src.translator import NllbTranslator
from src.vad import VadSegmenter


class TranslationPipeline:
    """Wire audio capture, segmentation, ASR and translation together."""

    def __init__(
        self,
        device: object,
        display: SubtitleDisplay,
        streaming: bool = False,
        backend: str = "local",
    ) -> None:
        self.device = device
        self.display = display
        self.streaming = streaming
        self.backend = backend
        self.cloud = backend != "local"
        self.stop_event = threading.Event()

        self._audio_queue: queue.Queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_MAXSIZE)
        # Bounded so a slow translator cannot accumulate unbounded backlog.
        self._text_queue: queue.Queue = queue.Queue(maxsize=32)
        # Makes the drop-oldest get+put on the text queue atomic vs. the consumer.
        self._text_queue_lock = threading.Lock()

        self.asr = None
        self.segmenter = None
        self.streaming_asr = None
        self.aggregator = None
        self._last_stream_text_at = monotonic()
        self.translator = None
        self.cloud_translator = None

        if self.cloud:
            self.display.info(f"Connecting to cloud backend ({backend})...")
            translator_cls = self._cloud_translator_class(backend)
            self.cloud_translator = translator_cls(self.display, on_fatal=self._on_cloud_fatal)
            self.display.info("Cloud backend ready.")
            self._capture = AudioCapture(self.device, self._audio_queue, self.stop_event)
            self._asr_thread = threading.Thread(target=self._cloud_worker, daemon=True)
            self._translate_thread = None
            return

        self.display.info("Loading models (ASR + translator)...")
        self.translator = NllbTranslator()
        if self.streaming:
            from src.streaming_asr import StreamingJapaneseASR

            self.streaming_asr = StreamingJapaneseASR()
            self.aggregator = SentenceAggregator()
            self._last_stream_text_at = monotonic()
        else:
            self.asr = JapaneseASR()
            self.segmenter = VadSegmenter()
        self.display.info("Models ready.")

        self._capture = AudioCapture(self.device, self._audio_queue, self.stop_event)
        asr_target = self._streaming_asr_worker if self.streaming else self._asr_worker
        self._asr_thread = threading.Thread(target=asr_target, daemon=True)
        self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)

    @staticmethod
    def _cloud_translator_class(backend: str):
        if backend == "azure":
            from src.cloud_translator import AzureSpeechTranslator

            return AzureSpeechTranslator
        raise ValueError(f"Unknown cloud backend: {backend!r} (supported: azure)")

    def _cloud_worker(self) -> None:
        """Stream captured audio to the cloud translator.

        The cloud service performs both recognition and translation and drives
        the display directly via its own callbacks, so this worker only has to
        forward audio blocks. Recognition/translation latency is bounded by the
        service (~0.5-1s) rather than local CPU.
        """
        while not self.stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.cloud_translator.push(block)
            except Exception as exc:  # keep forwarding audio on a single bad block
                self.display.info(f"[Cloud error] {exc}")

    def _on_cloud_fatal(self, message: str) -> None:
        """Stop the pipeline when the cloud backend reports a fatal error.

        Called from the Azure SDK callback thread, so it must not join threads
        or close the recognizer itself — it only signals ``run_forever`` to exit,
        which performs the orderly shutdown on the main thread.
        """
        self.stop_event.set()

    # ─── Workers ─────────────────────────────────────────────────────────
    def _asr_worker(self) -> None:
        """Segment captured audio and transcribe completed Japanese utterances."""
        while not self.stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                for utterance in self.segmenter.push(block):
                    self._transcribe_and_enqueue(utterance)
            except Exception as exc:  # keep the pipeline alive on a single bad block
                self.display.info(f"[ASR error] {exc}")

    def _streaming_asr_worker(self) -> None:
        """Stream audio into the online recognizer, showing live partials.

        Live Japanese partials are rewritten in place as they are spoken. Endpoint
        text is sentence-aggregated before translation so committed JP/VI pairs are
        printed atomically and in order.
        """
        agreement = LocalAgreementBuffer(config.STREAMING_LOCAL_AGREEMENT_N)
        last_displayed: tuple[str, str] = ("", "")
        while not self.stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                try:
                    self._flush_streaming_aggregator_if_due()
                except Exception as exc:
                    self.display.info(f"[ASR error] {exc}")
                continue
            try:
                self.streaming_asr.accept(block)
                partial = self.streaming_asr.partial()
                if partial:
                    self._last_stream_text_at = monotonic()
                    committed, tail = agreement.update(partial)
                    if (committed, tail) != last_displayed:
                        self.display.show_source_partial(committed, tail)
                        last_displayed = (committed, tail)
                if self.streaming_asr.at_endpoint():
                    if partial:
                        for sentence in self.aggregator.add(partial):
                            self._last_stream_text_at = monotonic()
                            self._enqueue_text(sentence, pre_shown=False)
                    self.streaming_asr.reset()
                    agreement.reset()
                    last_displayed = ("", "")
                self._flush_streaming_aggregator_if_due()
            except Exception as exc:  # keep the pipeline alive on a single bad block
                self.display.info(f"[ASR error] {exc}")

    def _flush_streaming_aggregator_if_due(self) -> None:
        if not self.streaming or self.aggregator is None:
            return
        pending = self.aggregator.pending().strip()
        if not pending:
            return
        idle_sec = monotonic() - self._last_stream_text_at
        if (
            len(pending) <= config.STREAM_SENTENCE_MAX_CHARS
            and idle_sec <= config.STREAM_SENTENCE_MAX_WAIT_SEC
        ):
            return
        for sentence in self.aggregator.flush():
            self._last_stream_text_at = monotonic()
            self._enqueue_text(sentence, pre_shown=False)

    def _transcribe_and_enqueue(self, utterance) -> None:
        japanese = self.asr.transcribe(utterance).strip()
        if not japanese:
            return
        # Show the recognized Japanese immediately so the viewer sees what was
        # said without waiting for the (slower) translation stage to finish.
        self.display.show_source(japanese)
        self._enqueue_text(japanese, pre_shown=True)

    def _enqueue_text(self, japanese: str, pre_shown: bool) -> None:
        # Lock makes the drop-oldest get+put atomic against the consumer, so a
        # concurrent take cannot cause us to drop an extra (still-fresh) item.
        with self._text_queue_lock:
            try:
                self._text_queue.put_nowait((japanese, pre_shown))
            except queue.Full:
                # Drop oldest pending text so the freshest speech wins under load.
                try:
                    self._text_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._text_queue.put_nowait((japanese, pre_shown))
                except queue.Full:
                    pass

    def _translate_worker(self) -> None:
        """Translate queued Japanese text into Vietnamese and display it."""
        while not self.stop_event.is_set():
            try:
                japanese, pre_shown = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                vietnamese = self.translator.translate(japanese).strip()
            except Exception as exc:
                self.display.info(f"[Translate error] {exc}")
                continue
            if pre_shown:
                self.display.show_target(vietnamese or "(...)")
            else:
                self.display.show_pair(japanese, vietnamese or "(...)")

    # ─── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self.cloud:
            self.cloud_translator.start()
        self._capture.start()
        self._asr_thread.start()
        if self._translate_thread is not None:
            self._translate_thread.start()

    def run_forever(self) -> None:
        """Block until interrupted, surfacing capture errors if they occur."""
        interrupted = False
        try:
            self.start()
            while not self.stop_event.is_set():
                if self._capture.error is not None:
                    self.display.info(f"[Audio error] {self._capture.error}")
                    break
                self.stop_event.wait(0.3)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            # On Ctrl+C, skip the trailing flush so a blocking model call can't
            # hang shutdown; on a normal stop, flush to keep the last words.
            self.stop(flush_tail=not interrupted)

    def stop(self, flush_tail: bool = True) -> None:
        self.stop_event.set()

        if self.cloud:
            # Cloud path has no local models to flush; just stop forwarding and
            # close the recognizer (which flushes any trailing audio server-side).
            self._capture.join(timeout=2.0)
            self._asr_thread.join(timeout=2.0)
            self.cloud_translator.stop()
            return

        # Join workers first so the non-thread-safe segmenter/ASR/translator are
        # no longer in use before the main thread touches them below. The ASR and
        # translate workers may be mid-inference in native (sherpa-onnx /
        # CTranslate2) code; give them a generous timeout so that call finishes
        # before the interpreter tears the native libraries down — otherwise a
        # still-running native call during shutdown can segfault the process.
        self._capture.join(timeout=2.0)
        self._asr_thread.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT)
        self._translate_thread.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT)

        if not flush_tail:
            return
        # If a worker is still running (join timed out mid-inference), skip the
        # trailing flush rather than race on the non-thread-safe models.
        if self._asr_thread.is_alive() or self._translate_thread.is_alive():
            return

        # Flush any trailing in-progress utterance so the last words aren't lost.
        try:
            if self.streaming:
                self.streaming_asr.finalize_tail()
                tail_text = self.streaming_asr.partial().strip()
                sentences: list[str] = []
                if tail_text:
                    sentences.extend(self.aggregator.add(tail_text))
                sentences.extend(self.aggregator.flush())
                for japanese in sentences:
                    vietnamese = self.translator.translate(japanese).strip()
                    self.display.show_pair(japanese, vietnamese or "(...)")
            else:
                tail = self.segmenter.flush()
                japanese = self.asr.transcribe(tail).strip() if tail is not None else ""
                if japanese:
                    vietnamese = self.translator.translate(japanese).strip()
                    self.display.show(japanese, vietnamese or "(...)")
        except Exception:
            pass
