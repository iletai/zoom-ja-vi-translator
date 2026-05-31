"""Structured per-stage evidence logging for debugging dropped data.

A real-time meeting translator loses content in subtle places: a bounded queue
drops the oldest item under load, a dedup filter skips a line, an aggregator
buffers a fragment that never flushes, or NLLB silently truncates a multi-
sentence block. None of these are visible from the terminal subtitles alone, so
a long meeting that "lost a sentence somewhere" is impossible to debug.

This module writes one JSON object per line (JSONL) for every pipeline event,
each carrying correlation fields (``seq``, monotonic timestamp, thread) so the
full life of a recognized utterance can be traced end to end:

    asr_final / aggregator_emit  ->  enqueue  ->  translate  ->  display
                                       \-> queue_drop / dedup_skip (loss!)

It is **opt-in** and a no-op unless configured (env ``ZT_EVIDENCE_LOG=<path>``
or ``--log <path>``). Events are sentence-level (a few per second at most), so a
synchronous, lock-guarded append adds negligible latency and never itself drops
events — correctness of the audit trail matters more here than micro-latency.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_fh = None  # type: ignore[assignment]
_path: Optional[Path] = None
_start_mono = time.monotonic()


def configure(path: Optional[str]) -> Optional[Path]:
    """Enable evidence logging to ``path`` (JSONL). ``None``/empty disables it.

    Falls back to the ``ZT_EVIDENCE_LOG`` environment variable when ``path`` is
    not given. Returns the resolved path, or ``None`` if logging stays disabled.
    Safe to call once at startup; reconfiguring closes the previous file.
    """
    global _fh, _path
    resolved = path or os.environ.get("ZT_EVIDENCE_LOG") or ""
    with _lock:
        if _fh is not None:
            try:
                _fh.close()
            finally:
                _fh = None
                _path = None
        if not resolved:
            return None
        target = Path(resolved).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append so each event hits disk immediately; a crash mid-
        # meeting still leaves a complete trail up to the last event.
        _fh = open(target, "a", encoding="utf-8", buffering=1)
        _path = target
    log("session_start", pid=os.getpid())
    return target


def is_enabled() -> bool:
    return _fh is not None


def log(event: str, **fields: Any) -> None:
    """Append one structured event. No-op when logging is disabled.

    Never raises into the pipeline: a logging failure must not take down audio.
    """
    if _fh is None:
        return
    record = {
        "ts": time.strftime("%H:%M:%S"),
        "t_ms": round((time.monotonic() - _start_mono) * 1000.0, 1),
        "thread": threading.current_thread().name,
        "event": event,
    }
    record.update(fields)
    try:
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            if _fh is not None:
                _fh.write(line + "\n")
    except Exception:  # pragma: no cover - logging must never break the pipeline
        pass


def close() -> None:
    """Flush and close the log file (best effort)."""
    global _fh, _path
    with _lock:
        if _fh is not None:
            try:
                _fh.close()
            finally:
                _fh = None
                _path = None
