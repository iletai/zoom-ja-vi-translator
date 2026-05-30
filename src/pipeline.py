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

import config
from src.asr import JapaneseASR
from src.audio_capture import AudioCapture
from src.display import SubtitleDisplay
from src.translator import NllbTranslator
from src.vad import VadSegmenter


class TranslationPipeline:
    """Wire audio capture, segmentation, ASR and translation together."""

    def __init__(self, device: object, display: SubtitleDisplay, streaming: bool = False) -> None:
        self.device = device
        self.display = display
        self.streaming = streaming
        self.stop_event = threading.Event()

        self._audio_queue: queue.Queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_MAXSIZE)
        # Bounded so a slow translator cannot accumulate unbounded backlog.
        self._text_queue: queue.Queue = queue.Queue(maxsize=32)
        # Makes the drop-oldest get+put on the text queue atomic vs. the consumer.
        self._text_queue_lock = threading.Lock()

        self.display.info("Loading models (ASR + translator)...")
        self.translator = NllbTranslator()
        if self.streaming:
            from src.streaming_asr import StreamingJapaneseASR

            self.asr = None
            self.segmenter = None
            self.streaming_asr = StreamingJapaneseASR()
        else:
            self.asr = JapaneseASR()
            self.segmenter = VadSegmenter()
            self.streaming_asr = None
        self.display.info("Models ready.")

        self._capture = AudioCapture(self.device, self._audio_queue, self.stop_event)
        asr_target = self._streaming_asr_worker if self.streaming else self._asr_worker
        self._asr_thread = threading.Thread(target=asr_target, daemon=True)
        self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)

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

        The recognized Japanese is rewritten in place as it is spoken; when the
        recognizer detects an endpoint the segment is committed and handed to the
        translator. This keeps the source text latency at roughly one audio block.
        """
        last_partial = ""
        while not self.stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.streaming_asr.accept(block)
                partial = self.streaming_asr.partial()
                if partial and partial != last_partial:
                    self.display.show_source_partial(partial)
                    last_partial = partial
                if self.streaming_asr.at_endpoint():
                    if partial:
                        self.display.finalize_source(partial)
                        self._enqueue_text(partial)
                    self.streaming_asr.reset()
                    last_partial = ""
            except Exception as exc:  # keep the pipeline alive on a single bad block
                self.display.info(f"[ASR error] {exc}")

    def _transcribe_and_enqueue(self, utterance) -> None:
        japanese = self.asr.transcribe(utterance).strip()
        if not japanese:
            return
        # Show the recognized Japanese immediately so the viewer sees what was
        # said without waiting for the (slower) translation stage to finish.
        self.display.show_source(japanese)
        self._enqueue_text(japanese)

    def _enqueue_text(self, japanese: str) -> None:
        # Lock makes the drop-oldest get+put atomic against the consumer, so a
        # concurrent take cannot cause us to drop an extra (still-fresh) item.
        with self._text_queue_lock:
            try:
                self._text_queue.put_nowait(japanese)
            except queue.Full:
                # Drop oldest pending text so the freshest speech wins under load.
                try:
                    self._text_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._text_queue.put_nowait(japanese)
                except queue.Full:
                    pass

    def _translate_worker(self) -> None:
        """Translate queued Japanese text into Vietnamese and display it."""
        while not self.stop_event.is_set():
            try:
                japanese = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                vietnamese = self.translator.translate(japanese).strip()
            except Exception as exc:
                self.display.info(f"[Translate error] {exc}")
                continue
            # The Japanese line was already shown by the ASR stage; print only the
            # Vietnamese line so the pair reads naturally without re-printing JP.
            self.display.show_target(vietnamese or "(...)")

    # ─── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        self._capture.start()
        self._asr_thread.start()
        self._translate_thread.start()

    def run_forever(self) -> None:
        """Block until interrupted, surfacing capture errors if they occur."""
        interrupted = False
        self.start()
        try:
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
                japanese = self.streaming_asr.partial().strip()
            else:
                tail = self.segmenter.flush()
                japanese = self.asr.transcribe(tail).strip() if tail is not None else ""
            if japanese:
                vietnamese = self.translator.translate(japanese).strip()
                self.display.show(japanese, vietnamese or "(...)")
        except Exception:
            pass
