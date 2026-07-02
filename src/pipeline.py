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

import logging
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import config
from src import evidence_log as ev
from src.asr import JapaneseASR
from src.audio_capture import AudioCapture
from src.display import SubtitleDisplay
from src.local_agreement import LocalAgreementBuffer
from src.post_correction import post_correct
from src.sentence_aggregator import SentenceAggregator, split_japanese_sentences
from src.translator import NllbTranslator, join_translations
# LLM translator is imported lazily to avoid ImportError when llama-cpp-python
# is not installed and the user is using the default NLLB backend.
from src.vad import VadSegmenter

logger = logging.getLogger(__name__)


# Sentinel pushed onto the segment queue to tell the offline re-decode worker the
# stream has ended and it should drain remaining utterances, then exit.
_SEGMENT_SENTINEL = object()


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
        self._session_id = str(uuid.uuid4())

        self._audio_queue: queue.Queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_MAXSIZE)
        # Bounded but generously sized: recognized text is preserved via producer
        # backpressure rather than dropped (dropping = permanent content loss).
        self._text_queue: queue.Queue = queue.Queue(maxsize=config.TEXT_QUEUE_MAXSIZE)
        # Monotonic id stamped on every recognized item so logs can trace one
        # utterance end-to-end and the translator can pair JP/VI by identity.
        self._seq = 0
        self._seq_lock = threading.Lock()

        self.asr = None
        self.segmenter = None
        self.streaming_asr = None
        self.aggregator = None
        self._last_stream_text_at = monotonic()
        self._last_enqueued_text = ""
        self._last_enqueued_at = 0.0
        # Re-join offline ASR fragments into whole sentences before translation so
        # NLLB receives context (see config.OFFLINE_AGGREGATE_SENTENCES). Touched by
        # exactly one worker (non-streaming _asr_worker XOR streaming _redecode_worker)
        # and, at shutdown, only by the main thread after that worker is joined — so
        # no locking is needed.
        self._offline_aggregator = None
        self._last_offline_text_at = monotonic()
        self.translator = None
        # Hybrid streaming: True = online model drives live partials only, while
        # a VAD-segmented offline re-decode produces the translated text.
        self._redecode = bool(self.streaming and config.STREAMING_REDECODE_OFFLINE)
        # Finalized raw-audio utterances handed from the online worker to the
        # offline re-decode worker. _SEGMENT_SENTINEL marks end-of-stream so the
        # re-decode worker drains everything (incl. the shutdown tail) before exit.
        self._segment_queue: queue.Queue = queue.Queue(maxsize=config.SEGMENT_QUEUE_MAX)
        self._redecode_thread = None
        # Allows _enqueue_text to keep enqueueing during an orderly shutdown drain
        # even after stop_event is set (so drained segments are never abandoned).
        self._draining = threading.Event()
        self.cloud_translator = None
        # ponytail: 2 workers — webhook is fire-and-forget HTTP; >2 concurrent posts are rare
        self._webhook_pool = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="webhook")
            if os.environ.get("ZT_WEBHOOK_URL")
            else None
        )

        if self.cloud:
            self.display.info(f"Connecting to cloud backend ({backend})...")
            logger.info("Initializing cloud backend: %s", backend)
            translator_cls = self._cloud_translator_class(backend)
            self.cloud_translator = translator_cls(self.display, on_fatal=self._on_cloud_fatal)
            self.display.info("Cloud backend ready.")
            logger.info("Cloud backend ready")
            self._capture = AudioCapture(self.device, self._audio_queue, self.stop_event)
            self._asr_thread = threading.Thread(target=self._cloud_worker, daemon=True)
            self._translate_thread = None
            return

        backend_label = {
            "llm": "LLM/Qwen2.5",
            "router": "Router/9router",
        }.get(config.TRANSLATOR_BACKEND, "NLLB")
        self.display.info(f"Loading models (ASR + {backend_label} translator)...")
        logger.info("Loading models: ASR + %s translator", backend_label)
        # Load ASR first (small ~160MB) before the heavier translator so that
        # on memory-constrained systems the combined peak stays below physical RAM.
        if self.streaming:
            from src.streaming_asr import StreamingJapaneseASR

            self.streaming_asr = StreamingJapaneseASR()
            self.aggregator = SentenceAggregator()
            self._last_stream_text_at = monotonic()
            if self._redecode:
                self.asr = JapaneseASR()
                self.segmenter = VadSegmenter()
        else:
            self.asr = JapaneseASR()
            self.segmenter = VadSegmenter()
        if self.asr is not None and config.OFFLINE_AGGREGATE_SENTENCES:
            self._offline_aggregator = SentenceAggregator()
        self.translator = self._create_translator()
        self.display.info(f"Models ready. Translator: {config.TRANSLATOR_BACKEND.upper()}")

        self._capture = AudioCapture(self.device, self._audio_queue, self.stop_event)
        asr_target = self._streaming_asr_worker if self.streaming else self._asr_worker
        self._asr_thread = threading.Thread(target=asr_target, daemon=True)
        self._translate_thread = threading.Thread(target=self._translate_worker, daemon=True)
        if self._redecode:
            self._redecode_thread = threading.Thread(target=self._redecode_worker, daemon=True)

    @staticmethod
    def _cloud_translator_class(backend: str):
        if backend == "azure":
            from src.cloud_translator import AzureSpeechTranslator

            return AzureSpeechTranslator
        raise ValueError(f"Unknown cloud backend: {backend!r} (supported: azure)")

    def _create_translator(self):
        """Create the translation engine based on config.TRANSLATOR_BACKEND."""
        if config.TRANSLATOR_BACKEND == "router":
            try:
                from src.router_translator import RouterTranslator

                return RouterTranslator()
            except ImportError as e:
                self.display.info(
                    f"[Warning] requests not installed, falling back to NLLB: {e}"
                )
                return NllbTranslator()
            except Exception as e:  # gateway misconfig/unreachable at init
                self.display.info(
                    f"[Warning] router backend init failed, falling back to NLLB: {e}"
                )
                return NllbTranslator()
        if config.TRANSLATOR_BACKEND == "llm":
            try:
                from src.llm_translator import LlmTranslator

                return LlmTranslator()
            except ImportError as e:
                self.display.info(
                    f"[Warning] llama-cpp-python not installed, falling back to NLLB: {e}"
                )
                return NllbTranslator()
            except FileNotFoundError as e:
                self.display.info(
                    f"[Warning] LLM model not found, falling back to NLLB: {e}"
                )
                return NllbTranslator()
        return NllbTranslator()

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

    def _on_cloud_fatal(self, _message: str) -> None:
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
                # No audio arriving: flush a buffered sentence that has gone idle.
                self._flush_offline_aggregator_if_due()
                continue
            try:
                for utterance in self.segmenter.push(block):
                    self._transcribe_and_enqueue(utterance)
                # Non-speech blocks keep this loop busy without ever hitting the
                # Empty branch, so also check the idle flush after each block.
                self._flush_offline_aggregator_if_due()
            except Exception as exc:  # keep the pipeline alive on a single bad block
                logger.error("ASR worker error: %s", exc, exc_info=True)
                self.display.info(f"[ASR error] {exc}")

    def _streaming_asr_worker(self) -> None:
        """Stream audio into the online recognizer, showing live partials.

        Live Japanese partials are rewritten in place as they are spoken. In the
        default hybrid mode the online model is used ONLY for the live partial
        display: completed utterances are segmented from the raw audio by VAD and
        handed to the offline re-decode worker, which produces the accurate text
        that is actually translated. With ZT_NO_REDECODE=1 the legacy behaviour is
        used: the online endpoint text itself is sentence-aggregated and translated.
        """
        agreement = LocalAgreementBuffer(config.STREAMING_LOCAL_AGREEMENT_N)
        last_displayed: tuple[str, str] = ("", "")
        try:
            while not self.stop_event.is_set():
                try:
                    block = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if not self._redecode:
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

                    if self._redecode:
                        # Offline re-decode path: VAD owns the segment boundaries
                        # (full audio incl. onsets) so the offline model never sees
                        # a clipped head. The online model only drives the display.
                        for utterance in self.segmenter.push(block):
                            self._put_segment(utterance)

                    if self.streaming_asr.at_endpoint():
                        if not self._redecode and partial:
                            for sentence in self.aggregator.add(partial):
                                self._last_stream_text_at = monotonic()
                                ev.log("aggregator_emit", text=sentence, reason="endpoint")
                                self._enqueue_text(sentence, pre_shown=False)
                        # Reset the online stream so the partial caption restarts
                        # cleanly at each utterance (display only in redecode mode).
                        self.streaming_asr.reset()
                        agreement.reset()
                        last_displayed = ("", "")
                    if not self._redecode:
                        self._flush_streaming_aggregator_if_due()
                except Exception as exc:  # keep the pipeline alive on a single bad block
                    logger.error("Streaming ASR worker error: %s", exc, exc_info=True)
                    self.display.info(f"[ASR error] {exc}")
        finally:
            if self._redecode:
                # Flush any trailing in-progress utterance, then signal the
                # re-decode worker to drain and exit (so the last words survive).
                try:
                    tail = self.segmenter.flush()
                    if tail is not None:
                        self._put_segment(tail)
                except Exception as exc:
                    ev.log("segment_flush_error", error=str(exc))
                # Sentinel must reach the consumer; block with timeouts (the
                # re-decode worker is still draining at this point) but never wedge
                # forever if the queue is wedged and the consumer is gone.
                while not self._segment_queue_put_sentinel():
                    if not (self._redecode_thread and self._redecode_thread.is_alive()):
                        break

    def _segment_queue_put_sentinel(self) -> bool:
        try:
            self._segment_queue.put(_SEGMENT_SENTINEL, timeout=0.5)
            return True
        except queue.Full:
            return False

    def _put_segment(self, utterance) -> None:
        """Hand a finalized raw-audio utterance to the offline re-decode worker.

        Blocks (with backpressure logging) rather than dropping: the audio is
        already-captured speech and dropping it is permanent content loss. The
        queue is generously sized, so this only blocks under sustained CPU
        overload, where stalling the online display is preferable to losing words.
        """
        duration_s = round(len(utterance) / float(config.SAMPLE_RATE), 2)
        # Diagnose WHY the segment ended: a duration at the max bound means the
        # silence endpoint never fired (VAD treated the whole window as voiced) and
        # the utterance was force-cut mid-speech — the signal to raise VAD
        # aggressiveness. Surfaced in the evidence log so this is visible per run.
        max_s = config.VAD_MAX_UTTERANCE_MS / 1000.0
        reason = "max_utterance" if duration_s >= max_s - (config.VAD_SILENCE_MS / 1000.0) else "silence"
        while not self.stop_event.is_set() or self._draining.is_set():
            # If draining is set but the redecode consumer is already dead, the
            # segment will never be consumed — abandon it and let the caller
            # know via the log event below so loss is never silent.
            if self._draining.is_set() and self._redecode_thread is not None and not self._redecode_thread.is_alive():
                break
            try:
                self._segment_queue.put(utterance, timeout=0.5)
                ev.log("segment_finalized", duration_s=duration_s,
                       reason=reason, queue_size=self._segment_queue.qsize())
                return
            except queue.Full:
                ev.log("segment_queue_backpressure", duration_s=duration_s)
        # Shutdown raced a full segment queue with no draining consumer: record
        # the abandoned utterance so loss is never silent.
        ev.log("segment_abandoned", duration_s=duration_s, reason="stop_event")

    def _redecode_worker(self) -> None:
        """Re-decode each finalized utterance's raw audio with the offline model.

        Owns the offline ASR + the aggregate/enqueue path so that the translated
        text comes from the strong model and ordering is preserved (single FIFO
        consumer). Runs until the end-of-stream sentinel, draining every queued
        utterance even after stop_event is set (shutdown drain)."""
        while True:
            try:
                item = self._segment_queue.get(timeout=0.3)
            except queue.Empty:
                # No new utterance: flush a buffered sentence that has gone idle.
                self._flush_offline_aggregator_if_due()
                continue
            if item is _SEGMENT_SENTINEL:
                # End of stream: emit any buffered partial sentence (force, so a
                # dangling tail is never silently dropped) before exiting.
                self._flush_offline_aggregator_if_due(force=True)
                return
            try:
                self._transcribe_and_enqueue(item)
            except Exception as exc:  # one bad utterance must not kill the worker
                ev.log("redecode_error", error=str(exc))

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
        # Don't emit a low-content dangling fragment (が/かと/から/…) on its own;
        # keep it buffered so it merges with the next utterance instead of
        # producing garbage like 「かと」 -> "Không, không.".
        if self.aggregator.is_dangling(pending):
            return
        for sentence in self.aggregator.flush():
            self._last_stream_text_at = monotonic()
            ev.log("aggregator_emit", text=sentence, reason="flush")
            self._enqueue_text(sentence, pre_shown=False)

    def _transcribe_and_enqueue(self, utterance) -> None:
        japanese = self.asr.transcribe(utterance).strip()
        if not japanese:
            return
        japanese = post_correct(japanese)
        logger.debug("ASR final: %s (%d chars)", japanese, len(japanese))
        ev.log("asr_final", text=japanese, n_chars=len(japanese))
        if self._offline_aggregator is None:
            # Legacy: translate each VAD segment independently. Show the recognized
            # Japanese immediately so the viewer sees what was said without waiting
            # for the (slower) translation stage; the seq ties this source line to
            # its later target line so batching keeps the JP/VI pair together.
            seq = self._next_seq()
            self.display.show_source(japanese, seq=seq)
            self._enqueue_text(japanese, pre_shown=True, seq=seq)
            return
        # VAD cuts on acoustic pauses, not sentence boundaries, so a single spoken
        # sentence arrives as several fragments. Re-join them into whole sentences
        # before translating: NLLB is sentence-trained and hallucinates on sub-
        # sentence fragments. The incomplete tail stays buffered until a terminal
        # arrives or the idle/shutdown flush emits it (never dropped).
        self._last_offline_text_at = monotonic()
        for sentence in self._offline_aggregator.add(japanese):
            ev.log("aggregator_emit", text=sentence, reason="boundary")
            self._emit_offline_sentence(sentence)

    def _emit_offline_sentence(self, japanese: str) -> None:
        """Display + enqueue one aggregated Japanese sentence (offline path).

        The seq pairs this committed source line with its later translation so the
        JP/VI pair stays together through batching.
        """
        seq = self._next_seq()
        self.display.show_source(japanese, seq=seq)
        self._enqueue_text(japanese, pre_shown=True, seq=seq)

    def _flush_offline_aggregator_if_due(self, force: bool = False) -> None:
        """Emit the buffered offline sentence when idle, oversized, or at shutdown.

        ``force`` (end-of-stream / shutdown) emits everything, including a low-
        content dangling tail — losing recognized text would violate the never-
        drop-data guarantee. The non-forced idle path keeps a dangling fragment
        (が/から/…) buffered so it merges with the next utterance instead of
        translating to garbage on its own.
        """
        if self._offline_aggregator is None:
            return
        pending = self._offline_aggregator.pending().strip()
        if not pending:
            return
        if not force:
            idle_sec = monotonic() - self._last_offline_text_at
            if (
                len(pending) <= config.STREAM_SENTENCE_MAX_CHARS
                and idle_sec <= config.OFFLINE_SENTENCE_MAX_WAIT_SEC
            ):
                return
            # Give incomplete fragments (ending in connective particles) extra
            # buffer time — the continuation may arrive in the next VAD segment.
            if (
                idle_sec <= config.OFFLINE_SENTENCE_MAX_WAIT_SEC * 2
                and self._offline_aggregator.ends_with_connective(pending)
            ):
                return
            if self._offline_aggregator.is_dangling(pending):
                return
        for sentence in self._offline_aggregator.flush():
            self._last_offline_text_at = monotonic()
            ev.log("aggregator_emit", text=sentence,
                   reason="shutdown" if force else "flush")
            self._emit_offline_sentence(sentence)

    def _log_offline_aggregator_abandoned(self, reason: str) -> None:
        """Record buffered offline text that a non-flushing shutdown abandons.

        Aggregated text can sit only in the aggregator's buffer (not in any queue),
        so the queue-size accounting would otherwise miss it. Logging keeps the
        never-silently-lost-data audit trail complete even when we deliberately
        skip the trailing flush (e.g. Ctrl+C, or a worker stuck mid-inference)."""
        if self._offline_aggregator is None:
            return
        pending = self._offline_aggregator.pending().strip()
        if pending:
            ev.log("offline_aggregator_abandoned", text=pending,
                   n_chars=len(pending), reason=reason)

    def _segment_queue_pending_count(self) -> int:
        # ponytail: qsize() is O(1) and lock-free; sentinel inflates by at most 1 — fine for shutdown logging
        return self._segment_queue.qsize()

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _enqueue_text(self, japanese: str, pre_shown: bool, seq: int | None = None) -> None:
        if not pre_shown and config.STREAM_DEDUP_ENABLED:
            stripped = japanese.strip()
            now = monotonic()
            if (
                stripped == self._last_enqueued_text
                and len(stripped) >= config.STREAM_DEDUP_MIN_CHARS
                and now - self._last_enqueued_at <= config.STREAM_DEDUP_WINDOW_SEC
            ):
                ev.log("dedup_skip", text=stripped, n_chars=len(stripped))
                return
            self._last_enqueued_text = stripped
            self._last_enqueued_at = now

        if seq is None:
            seq = self._next_seq()
        item = (seq, japanese, pre_shown)
        ev.log(
            "enqueue",
            seq=seq,
            text=japanese,
            pre_shown=pre_shown,
            queue_size=self._text_queue.qsize(),
        )
        # Recognized text is already-captured meeting content; dropping it loses
        # the user's words permanently. Apply backpressure (block, re-checking
        # the stop flag) instead of drop-oldest so nothing recognized is lost.
        # Under sustained overload this stalls the ASR worker, letting the *audio*
        # queue shed raw blocks instead — a far smaller content loss. During an
        # orderly shutdown drain (_draining) we keep enqueueing past stop_event so
        # the re-decode worker's drained tail utterances are never abandoned.
        while not self.stop_event.is_set() or self._draining.is_set():
            # If the translate thread is dead, text can never be consumed —
            # abandon the item to prevent the ASR worker from blocking forever.
            if self._translate_thread is not None and not self._translate_thread.is_alive():
                ev.log("enqueue_abandoned", seq=seq, text=japanese, reason="translate_thread_dead")
                return
            try:
                self._text_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                ev.log("text_queue_backpressure", seq=seq, queue_size=self._text_queue.qsize())
                continue
        # Shutdown raced an overloaded queue: record the abandoned item so the
        # audit trail never has a silent seq gap (stop() drains the queue, but an
        # item still blocked here was never enqueued).
        ev.log("enqueue_abandoned", seq=seq, text=japanese, reason="stop_event")


    def _translate_worker(self) -> None:
        """Translate queued Japanese text into Vietnamese and display it.

        Drains a batch of pending items and translates them together so a fast
        meeting cannot build an unbounded backlog. Items are processed in FIFO
        order and displayed by identity, so JP/VI pairs never cross.
        """
        while not self.stop_event.is_set() or self._draining.is_set():
            try:
                try:
                    first = self._text_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                batch = [first]
                while len(batch) < config.TRANSLATE_MAX_BATCH:
                    try:
                        batch.append(self._text_queue.get_nowait())
                    except queue.Empty:
                        break
                try:
                    self._translate_and_display_batch(batch)
                except Exception as exc:  # keep the translate thread alive
                    logger.error("Translate batch error: %s", exc, exc_info=True)
                    self.display.info(f"[Translate error] {exc}")
                    for seq, jp, _ in batch:
                        ev.log("translate_error", seq=seq, text=jp, error=str(exc))
            except Exception as exc:
                # Safety net: any unhandled exception (including the queue.Empty
                # or batch-getting logic) must not kill the translate thread.
                # A dead translate thread means _enqueue_text blocks forever
                # and the entire pipeline freezes.
                logger.error("Translate worker inner error: %s", exc, exc_info=True)
                continue

    def _translate_and_display_batch(self, batch: list[tuple[int, str, bool]]) -> None:
        started = monotonic()
        logger.debug("Translating batch of %d item(s)", len(batch))
        try:
            translations = self._translate_batch_flattened(batch)
        except Exception as exc:
            # Fall back to per-item translation so one bad item cannot drop the
            # whole batch; a still-failing item is logged and shown as "(...)".
            logger.error("Batch translation failed, falling back to per-item: %s", exc, exc_info=True)
            self.display.info(f"[Translate error] {exc}")
            translations = []
            for seq, jp, _ in batch:
                try:
                    translations.append(self.translator.translate(jp).strip())
                except Exception as item_exc:
                    logger.error("Per-item translate failed seq=%d: %s", seq, item_exc)
                    ev.log("translate_error", seq=seq, text=jp, error=str(item_exc))
                    translations.append("")
        # Guard against any translation/source count mismatch: zip() would
        # silently drop the unpaired tail items (lost content). Pad short and
        # never truncate the batch.
        if len(translations) < len(batch):
            for seq, jp, _ in batch[len(translations):]:
                ev.log("translate_count_mismatch", seq=seq, text=jp,
                       got=len(translations), expected=len(batch))
            translations = translations + [""] * (len(batch) - len(translations))
        latency_ms = round((monotonic() - started) * 1000.0, 1)
        logger.info("Translated batch(%d) in %.1fms", len(batch), latency_ms)

        for (seq, japanese, pre_shown), vietnamese in zip(batch, translations):
            logger.debug("seq=%d JP: %s -> VI: %s", seq, japanese, vietnamese)
            ev.log(
                "translate",
                seq=seq,
                jp=japanese,
                vi=vietnamese,
                latency_ms=latency_ms,
                batch=len(batch),
            )
            # Isolate display/log failures per item so one bad render cannot drop
            # the rest of an already-translated batch.
            try:
                if pre_shown:
                    self.display.show_target(vietnamese or "(...)", japanese=japanese, seq=seq)
                else:
                    self.display.show_pair(japanese, vietnamese or "(...)")
                ev.log("display", seq=seq, pre_shown=pre_shown, jp=japanese, vi=vietnamese)
            except Exception as exc:
                ev.log("display_error", seq=seq, jp=japanese, vi=vietnamese, error=str(exc))
            if vietnamese:
                self._fire_webhook(seq, japanese, vietnamese)

    def _fire_webhook(self, seq: int, japanese: str, vietnamese: str) -> None:
        """POST one translated pair to the webhook (fire-and-forget).

        Sends thread_id from the first response so the Flow can route subsequent
        messages as replies in-thread instead of new posts.
        """
        if self._webhook_pool is None:
            return
        url = os.environ.get("ZT_WEBHOOK_URL", "")
        if not url:
            return
        timeout = float(os.environ.get("ZT_WEBHOOK_TIMEOUT", "12"))
        proxy_url = os.environ.get("ZT_WEBHOOK_PROXY", "")
        proxies = {"https": proxy_url, "http": proxy_url} if proxy_url else {}

        payload = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": f"🎙️ [{seq}] {japanese}",
                    "wrap": True,
                    "size": "Small",
                    "color": "Accent",
                },
                {
                    "type": "TextBlock",
                    "text": f"🇻🇳 {vietnamese}",
                    "wrap": True,
                    "size": "Default",
                    "weight": "Bolder",
                },
            ],
        }

        def _post() -> None:
            try:
                import requests as _req
                s = _req.Session()
                s.trust_env = not proxy_url  # False = ignore HTTPS_PROXY from Windows env
                if proxy_url:
                    s.proxies = proxies
                s.post(url, json=payload, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Webhook POST failed seq=%d: %s", seq, exc)

        self._webhook_pool.submit(_post)

    def _translate_batch_flattened(self, batch: list[tuple[int, str, bool]]) -> list[str]:
        """Translate every sentence across all items in one CTranslate2 batch.

        Each item may itself be multiple sentences (NLLB drops trailing ones if
        not split). We split every item, translate the flattened sentence list in
        a single native call so CTranslate2 parallelizes across the whole batch —
        the throughput that clears backlog fast — then rejoin per item in order.
        """
        per_item_sentences = [split_japanese_sentences(jp) or [jp] for _, jp, _ in batch]
        flat = [s for sentences in per_item_sentences for s in sentences]
        if not flat:
            return ["" for _ in batch]
        flat_vi = self.translator.translate_many(flat)
        # ponytail: translate_many shouldn't return fewer items, but if it does
        # a short slice shifts every subsequent item's alignment silently.
        if len(flat_vi) < len(flat):
            ev.log("translate_many_short", got=len(flat_vi), expected=len(flat))
            flat_vi = flat_vi + [""] * (len(flat) - len(flat_vi))

        out: list[str] = []
        cursor = 0
        for item, sentences in zip(batch, per_item_sentences):
            count = len(sentences)
            joined, dropped = join_translations(
                sentences, flat_vi[cursor : cursor + count]
            )
            if dropped:
                seq = item[0]
                for idx, src in dropped:
                    ev.log("empty_translation", seq=seq, sentence_index=idx, jp=src)
            out.append(joined.strip())
            cursor += count
        return out

    def _translate_and_display_shutdown_sentence(self, japanese: str, pre_shown: bool) -> None:
        seq = self._next_seq()
        if pre_shown:
            try:
                self.display.show_source(japanese, seq=seq)
            except Exception as exc:
                ev.log("display_error", seq=seq, jp=japanese, vi="", error=str(exc))
        ev.log(
            "enqueue",
            seq=seq,
            text=japanese,
            pre_shown=pre_shown,
            queue_size=self._text_queue.qsize(),
        )
        started = monotonic()
        try:
            vietnamese = self.translator.translate(japanese).strip()
        except Exception as exc:
            ev.log("translate_error", seq=seq, text=japanese, error=str(exc))
            vietnamese = ""
        latency_ms = round((monotonic() - started) * 1000.0, 1)
        ev.log(
            "translate",
            seq=seq,
            jp=japanese,
            vi=vietnamese,
            latency_ms=latency_ms,
            batch=1,
        )
        try:
            if pre_shown:
                self.display.show_target(vietnamese or "(...)", japanese=japanese, seq=seq)
            else:
                self.display.show_pair(japanese, vietnamese or "(...)")
            ev.log("display", seq=seq, pre_shown=pre_shown, jp=japanese, vi=vietnamese)
        except Exception as exc:
            ev.log("display_error", seq=seq, jp=japanese, vi=vietnamese, error=str(exc))

    def _drain_text_queue(self, deadline: float | None = None) -> None:
        """Translate and display any items left in the text queue at shutdown.

        Called from the main thread in stop() AFTER the workers are joined, so the
        non-thread-safe translator/display are no longer used concurrently. Drains
        in TRANSLATE_MAX_BATCH-sized chunks (not one unbounded batch) to bound
        memory and native call size.
        """
        while True:
            if deadline is not None and monotonic() >= deadline:
                return
            batch: list[tuple[int, str, bool]] = []
            while len(batch) < config.TRANSLATE_MAX_BATCH:
                try:
                    batch.append(self._text_queue.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                return
            ev.log("shutdown_drain", count=len(batch))
            self._translate_and_display_batch(batch)

    # ─── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self.cloud:
            self.cloud_translator.start()
        self._capture.start()
        self._asr_thread.start()
        if self._redecode_thread is not None:
            self._redecode_thread.start()
        if self._translate_thread is not None:
            self._translate_thread.start()

    def run_forever(self) -> None:
        """Block until interrupted, surfacing capture errors if they occur."""
        from src.mem_guard import MemoryMonitor

        mem_monitor = MemoryMonitor(
            text_queue=self._text_queue,
            audio_queue=self._audio_queue,
        )
        mem_monitor.start()
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
            mem_monitor.stop()
            # On Ctrl+C, skip the trailing flush so a blocking model call can't
            # hang shutdown; on a normal stop, flush to keep the last words.
            self.stop(flush_tail=not interrupted)

    def stop(self, flush_tail: bool = True) -> None:
        if self._redecode:
            # Permit the re-decode worker's drained tail utterances to keep
            # enqueueing past stop_event during the orderly shutdown below.
            self._draining.set()
        self.stop_event.set()

        if self.cloud:
            # Cloud path has no local models to flush; just stop forwarding and
            # close the recognizer (which flushes any trailing audio server-side).
            self._capture.join(timeout=2.0)
            self._asr_thread.join(timeout=2.0)
            self.cloud_translator.stop()
            if self._webhook_pool is not None:
                self._webhook_pool.shutdown(wait=False)
            return

        # Join workers first so the non-thread-safe segmenter/ASR/translator are
        # no longer in use before the main thread touches them below. The ASR and
        # translate workers may be mid-inference in native (sherpa-onnx /
        # CTranslate2) code; give them a generous timeout so that call finishes
        # before the interpreter tears the native libraries down — otherwise a
        # still-running native call during shutdown can segfault the process.
        self._capture.join(timeout=2.0)
        # The online worker, on exit, flushes the VAD tail and pushes the
        # end-of-stream sentinel that lets the re-decode worker drain and exit.
        self._asr_thread.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT)
        if self._redecode_thread is not None:
            # Drain finalized utterances through the offline model. On Ctrl+C
            # (flush_tail=False) bound the wait so shutdown stays responsive. The
            # translate worker stays alive during this drain (its loop also runs
            # while _draining) so re-decoded text is consumed, not wedged in a
            # full _text_queue.
            redecode_timeout = config.WORKER_SHUTDOWN_TIMEOUT if flush_tail else 2.0
            self._redecode_thread.join(timeout=redecode_timeout)
            # Re-decode is done producing: let the translate worker exit.
            self._draining.clear()
        self._translate_thread.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT)

        if self._webhook_pool is not None:
            self._webhook_pool.shutdown(wait=False)

        # Release the translator's resources (Router's pooled HTTP session);
        # NLLB/LLM have no close(), so guard with getattr.
        close = getattr(self.translator, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                ev.log("translator_close_error", error=str(exc))

        if not flush_tail:
            # Ctrl+C best-effort: keep shutdown bounded, but audit and drain any
            # already-recognized text that is safe to process on this thread.
            text_pending = self._text_queue.qsize()
            segment_pending = self._segment_queue_pending_count()
            if text_pending or segment_pending:
                ev.log(
                    "shutdown_unflushed",
                    text_queue=text_pending,
                    segment_queue=segment_pending,
                    reason="ctrl_c",
                )
            if not self._translate_thread.is_alive():
                try:
                    self._drain_text_queue(deadline=monotonic() + 0.5)
                except Exception as exc:
                    ev.log("shutdown_drain_error", error=str(exc), reason="ctrl_c")
            else:
                ev.log("shutdown_drain_skipped", reason="ctrl_c", translate_thread_alive=True)
            self._log_offline_aggregator_abandoned("ctrl_c")
            return
        # If a worker is still running (join timed out mid-inference), skip the
        # trailing flush rather than race on the non-thread-safe models — but
        # surface how much recognized text is being abandoned so it is never a
        # silent loss.
        redecode_busy = self._redecode_thread is not None and self._redecode_thread.is_alive()
        if self._asr_thread.is_alive() or self._translate_thread.is_alive() or redecode_busy:
            pending = self._text_queue.qsize() + self._segment_queue_pending_count()
            agg_pending = (
                self._offline_aggregator.pending().strip()
                if self._offline_aggregator is not None
                else ""
            )
            if pending or agg_pending:
                ev.log("shutdown_unflushed", queue_size=pending,
                       aggregator_pending=agg_pending)
                detail = f"{pending} recognized item(s)"
                if agg_pending:
                    detail += f" + {len(agg_pending)} buffered char(s)"
                self.display.info(
                    f"[Shutdown] {detail} not translated (worker still busy)."
                )
            return

        # The translate worker has exited, so the text queue may still hold
        # recognized items it never drained. These are already-captured words —
        # translate and display them (on this thread, now that the worker is
        # joined and the models are idle) before flushing the in-progress tail.
        # Queued items are chronologically older than the in-progress tail, so
        # drain them first to keep on-screen order.
        try:
            self._drain_text_queue()
        except Exception as exc:
            ev.log("shutdown_drain_error", error=str(exc))

        # Flush any trailing in-progress utterance so the last words aren't lost.
        sentences: list[str] = []
        pending_before_flush = ""
        try:
            if self._redecode:
                # The VAD tail was already finalized into the segment queue and
                # translated via the re-decode worker drain above; nothing else
                # to flush (the online stream is display-only in this mode).
                pass
            elif self.streaming:
                self.streaming_asr.finalize_tail()
                tail_text = self.streaming_asr.partial().strip()
                if tail_text:
                    sentences.extend(self.aggregator.add(tail_text))
                pending_before_flush = self.aggregator.pending().strip()
                sentences.extend(self.aggregator.flush())
                for japanese in sentences:
                    self._translate_and_display_shutdown_sentence(japanese, pre_shown=False)
            else:
                tail = self.segmenter.flush()
                tail_text = self.asr.transcribe(tail).strip() if tail is not None else ""
                if tail_text:
                    tail_text = post_correct(tail_text)
                # Drain the offline sentence aggregator too: a buffered fragment
                # lives only in the aggregator (not the text queue), so a normal
                # stop must flush it or the last words are lost.
                if self._offline_aggregator is not None:
                    if tail_text:
                        sentences.extend(self._offline_aggregator.add(tail_text))
                    pending_before_flush = self._offline_aggregator.pending().strip()
                    sentences.extend(self._offline_aggregator.flush())
                elif tail_text:
                    sentences = [tail_text]
                    pending_before_flush = tail_text
                for japanese in sentences:
                    self._translate_and_display_shutdown_sentence(japanese, pre_shown=True)
        except Exception as exc:
            ev.log(
                "shutdown_tail_abandoned",
                reason="normal_stop",
                error=str(exc),
                aggregator_pending=pending_before_flush,
                sentences=sentences,
            )
