"""Memory guard for the real-time translation pipeline.

Provides:
  - MemoryMonitor: background thread that logs RSS periodically, warns on growth.
  - rss_mb(): current process resident set size.
  - oom_headroom_mb(): estimated available memory before OOM.
  - force_gc(): targeted GC collection between translation batches.
  - ct2_allocator_reset(): safe CTranslate2 allocator drain.

Usage in pipeline.py:
    from src.mem_guard import MemoryMonitor, force_gc
    monitor = MemoryMonitor()
    monitor.start()
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import queue as queue_mod

logger = logging.getLogger(__name__)

# psutil is optional — fall back to /proc on Linux, skip on Windows without it
try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def rss_mb() -> float:
    """Resident set size of the current process, in MB."""
    if _HAVE_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _total_ram_mb() -> float:
    if _HAVE_PSUTIL:
        return psutil.virtual_memory().total / (1024 ** 2)
    return 0.0


def oom_headroom_mb() -> float:
    """Approximate MB available before OOM (requires psutil)."""
    if not _HAVE_PSUTIL:
        return float("inf")
    vm = psutil.virtual_memory()
    return vm.available / (1024 ** 2)


def force_gc(generation: int = 1) -> int:
    """Run a targeted Python GC collection. Returns objects freed."""
    collected = gc.collect(generation)
    if collected:
        logger.debug("gc.collect(gen=%d) freed %d objects", generation, collected)
    return collected


def ct2_allocator_reset(translator) -> bool:
    """Drain CTranslate2's caching allocator by unloading and reloading the model.

    Returns True on success. This releases pooled C++ buffers accumulated since
    the last reset, at the cost of ~1-2s pause while weights reload.
    """
    unload = getattr(translator, "unload_model", None)
    load = getattr(translator, "load_model", None)
    if unload is None or load is None:
        return False
    try:
        logger.info("CT2 allocator reset: unloading model")
        unload(to_cpu=False)
        force_gc(generation=2)
        logger.info("CT2 allocator reset: reloading model")
        load()
        logger.info("CT2 allocator reset: complete — RSS now %.1f MB", rss_mb())
        return True
    except Exception as exc:
        logger.warning("CT2 allocator reset failed: %s", exc)
        return False


class MemoryMonitor(threading.Thread):
    """Background thread: log RSS + queue depths every interval_s seconds.

    Emits WARNING when RSS grows by more than warn_delta_mb from baseline,
    or when oom_headroom_mb() drops below critical_headroom_mb.
    """

    def __init__(
        self,
        interval_s: float = 30.0,
        warn_delta_mb: float = 300.0,
        critical_headroom_mb: float = 500.0,
        text_queue: "Optional[queue_mod.Queue]" = None,
        audio_queue: "Optional[queue_mod.Queue]" = None,
        segment_queue: "Optional[queue_mod.Queue]" = None,
    ):
        super().__init__(daemon=True, name="MemoryMonitor")
        self.interval_s = interval_s
        self.warn_delta_mb = warn_delta_mb
        self.critical_headroom_mb = critical_headroom_mb
        self._text_queue = text_queue
        self._audio_queue = audio_queue
        self._segment_queue = segment_queue
        self._stop = threading.Event()
        self._baseline_mb: Optional[float] = None

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._report()

    def _report(self) -> None:
        current_rss = rss_mb()
        headroom = oom_headroom_mb()

        if self._baseline_mb is None:
            self._baseline_mb = current_rss
            logger.info(
                "[MemGuard] Baseline RSS=%.1f MB | Total RAM=%.1f MB | Headroom=%.1f MB",
                current_rss, _total_ram_mb(), headroom,
            )
            return

        delta = current_rss - self._baseline_mb
        tq = self._text_queue.qsize() if self._text_queue else -1
        aq = self._audio_queue.qsize() if self._audio_queue else -1
        sq = self._segment_queue.qsize() if self._segment_queue else -1

        if headroom < self.critical_headroom_mb:
            logger.critical(
                "[MemGuard] CRITICAL: only %.1f MB headroom! RSS=%.1f MB Δ=%+.1f MB "
                "| text_q=%d audio_q=%d seg_q=%d",
                headroom, current_rss, delta, tq, aq, sq,
            )
        elif delta > self.warn_delta_mb:
            logger.warning(
                "[MemGuard] High growth: RSS=%.1f MB Δ=%+.1f MB headroom=%.1f MB "
                "| text_q=%d audio_q=%d seg_q=%d",
                current_rss, delta, headroom, tq, aq, sq,
            )
        else:
            logger.debug(
                "[MemGuard] RSS=%.1f MB Δ=%+.1f MB headroom=%.1f MB "
                "| text_q=%d audio_q=%d seg_q=%d",
                current_rss, delta, headroom, tq, aq, sq,
            )

    def stop(self) -> None:
        self._stop.set()
