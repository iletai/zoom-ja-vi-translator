"""Pure, Streamlit-free helpers for the live dashboard.

Kept separate from ``streamlit_app.py`` so the feed-shaping logic (search,
ordering, tail) can be unit-tested without importing Streamlit or starting a
server runtime.
"""
from __future__ import annotations

import re
from typing import Sequence, TypeVar

# ``TranscriptLine`` is a plain dataclass; we only touch ``.jp`` / ``.vi`` here,
# so a structural TypeVar keeps this module decoupled from transcript_export.
T = TypeVar("T")


def filter_lines(lines: Sequence[T], query: str) -> list[T]:
    """Return lines whose JP or VI text contains ``query`` (case-insensitive).

    An empty/blank query returns all lines (as a new list).
    """
    q = (query or "").strip().lower()
    if not q:
        return list(lines)
    return [ln for ln in lines if q in ln.jp.lower() or q in ln.vi.lower()]


def prepare_feed(
    lines: Sequence[T],
    query: str = "",
    tail: int = 80,
    newest_first: bool = False,
) -> list[T]:
    """Shape the feed for display: filter by query, keep the most recent
    ``tail`` lines (chronologically), then optionally reverse to newest-first.

    ``tail <= 0`` means "no limit". The most-recent slice is always taken in
    chronological order *before* reordering, so newest-first never changes
    *which* lines are shown — only their order.
    """
    filtered = filter_lines(lines, query)
    recent = filtered[-tail:] if tail and tail > 0 else list(filtered)
    if newest_first:
        recent = list(reversed(recent))
    return recent


def latency_alert(median_ms: float, threshold_ms: float) -> bool:
    """True when median latency exceeds a positive threshold (0/blank = off)."""
    return threshold_ms > 0 and median_ms > threshold_ms


def safe_stem(stem: str) -> str:
    """Sanitize a user-supplied filename stem.

    Strips path separators and unsafe characters so a download filename can
    never escape into a path; falls back to ``"transcript"`` when empty.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (stem or "").strip()).strip("._-")
    return cleaned or "transcript"
