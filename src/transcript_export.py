"""Build and export a bilingual transcript from a JSONL evidence log.

The pipeline writes a structured evidence log (see ``evidence_log.py``) with one
event per pipeline stage. This module reconstructs the final, ordered list of
translated subtitle lines from that log and renders it as plain text, Markdown,
SRT subtitles, or JSON — so a finished meeting can be saved and re-read.

Timing: subtitle windows are derived from the ``display`` events' wall-clock
``t_ms`` — each line spans from its own display time to the next line's display
time (clamped to a readable range). Sentence aggregation breaks the 1:1
segment↔display relationship, so timing is taken from the display timeline
rather than joined to ``segment_finalized`` events. This is robust for both
legacy logs (no per-segment seq) and aggregated runs, and never shifts when a
finalized segment produces no display.

Usage:
    python -m src.transcript_export run.jsonl --format srt -o run.srt
    python -m src.transcript_export run.jsonl --format md   # stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

_LOSS_EVENTS = {
    "audio_block_dropped",
    "audio_capture_error",
    "segment_abandoned",
    "empty_translation",
    "segment_queue_backpressure",
    "text_queue_backpressure",
    "redecode_error",
    "dedup_skip",
    "translate_count_mismatch",
    "display_error",
}


@dataclass
class TranscriptLine:
    """One finished bilingual subtitle line."""

    seq: int
    jp: str
    vi: str
    start_ms: float
    end_ms: float
    ts: str  # wall-clock HH:MM:SS of the display event


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL evidence log into a list of event dicts (bad lines skipped)."""
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# Bounds for a derived subtitle line duration (ms). A line spans from its own
# display time to the next line's display time, clamped into this range so a
# long pause before the next utterance doesn't leave one subtitle on screen
# forever and a rapid-fire pair still shows for a readable minimum.
_MIN_LINE_MS = 800.0
_MAX_LINE_MS = 8000.0
_LAST_LINE_MS = 3000.0


def build_lines(events: Iterable[dict[str, Any]]) -> list[TranscriptLine]:
    """Reconstruct the ordered bilingual transcript from evidence events.

    ``display`` events are canonical: each carries the final jp+vi+seq and the
    wall-clock ``t_ms`` at which the line was shown. Timing is derived purely
    from these display timestamps — each line spans from its own ``t_ms`` to the
    next line's ``t_ms`` (clamped) — rather than by joining to ``segment_finalized``
    events. Sentence aggregation deliberately breaks the 1:1 segment↔display
    relationship (one merged sentence can come from several VAD segments, and one
    run-on segment can yield several displays), so neither positional nor
    seq-based pairing with segments is well-defined; using the display timeline
    is robust for old and new logs alike and never shifts on a skipped display.

    Duplicate displays for the same ``seq`` (a pre-shown line later superseded by
    its final text) collapse to the last occurrence so the final wording wins.
    """
    latest_by_seq: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    fallback_seq = 0
    for e in events:
        if e.get("event") != "display":
            continue
        raw_seq = e.get("seq")
        if raw_seq is None:
            fallback_seq -= 1  # keep order, never collide with real (positive) seqs
            seq = fallback_seq
        else:
            seq = int(raw_seq)
        if seq not in latest_by_seq:
            order.append(seq)
        latest_by_seq[seq] = e

    # Order lines by display time (t_ms), falling back to seq for stability.
    ordered = sorted(order, key=lambda s: (float(latest_by_seq[s].get("t_ms", 0.0)), s))
    starts = [float(latest_by_seq[s].get("t_ms", 0.0)) for s in ordered]

    lines: list[TranscriptLine] = []
    prev_end = 0.0
    for i, seq in enumerate(ordered):
        disp = latest_by_seq[seq]
        start_ms = max(starts[i], prev_end)
        if i + 1 < len(ordered):
            gap = starts[i + 1] - starts[i]
            span = min(_MAX_LINE_MS, max(_MIN_LINE_MS, gap))
        else:
            span = _LAST_LINE_MS
        end_ms = start_ms + span
        prev_end = end_ms
        lines.append(
            TranscriptLine(
                seq=seq,
                jp=(disp.get("jp") or "").strip(),
                vi=(disp.get("vi") or "").strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                ts=str(disp.get("ts", "")),
            )
        )
    return lines


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute headline stats for a run: counts, reason split, loss, latency."""
    events = list(events)
    durations: list[float] = []
    reasons: dict[str, int] = {}
    latencies: list[float] = []
    loss: dict[str, int] = {}
    for e in events:
        ev = e.get("event")
        if ev == "segment_finalized":
            durations.append(float(e.get("duration_s", 0.0)))
            r = e.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        elif ev == "translate" and e.get("latency_ms") is not None:
            latencies.append(float(e["latency_ms"]))
        elif ev in _LOSS_EVENTS:
            loss[ev] = loss.get(ev, 0) + 1

    n_disp = sum(1 for e in events if e.get("event") == "display")
    return {
        "segments": len(durations),
        "displayed": n_disp,
        "reasons": reasons,
        "max_utterance_pct": round(
            100.0 * reasons.get("max_utterance", 0) / len(durations), 1
        )
        if durations
        else 0.0,
        "duration_median_s": round(_median(durations), 2),
        "duration_max_s": round(max(durations), 2) if durations else 0.0,
        "translate_latency_median_ms": round(_median(latencies), 1),
        "translate_latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
        "loss_events": loss,
    }


def _fmt_srt_ts(ms: float) -> str:
    """Format milliseconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    ms = max(0, int(round(ms)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(lines: list[TranscriptLine], bilingual: bool = True) -> str:
    """Render lines as an SRT subtitle file (VI on top, JP below when bilingual)."""
    blocks: list[str] = []
    previous_end = 0.0
    for i, ln in enumerate(lines, start=1):
        start = max(previous_end, ln.start_ms)
        end = ln.end_ms if ln.end_ms > start else start + 1500.0
        text = ln.vi if not bilingual else f"{ln.vi}\n{ln.jp}".strip()
        blocks.append(f"{i}\n{_fmt_srt_ts(start)} --> {_fmt_srt_ts(end)}\n{text}\n")
        previous_end = end
    return "\n".join(blocks)


def to_text(lines: list[TranscriptLine]) -> str:
    """Render lines as a plain bilingual transcript."""
    out: list[str] = []
    for ln in lines:
        stamp = ln.ts or _fmt_srt_ts(ln.start_ms)
        out.append(f"[{stamp}]")
        out.append(f"  JP {ln.jp}")
        out.append(f"  VI {ln.vi}")
        out.append("")
    return "\n".join(out)


def to_markdown(lines: list[TranscriptLine]) -> str:
    """Render lines as a Markdown table."""
    out = ["| # | Time | Japanese | Vietnamese |", "|---|---|---|---|"]
    for ln in lines:
        jp = ln.jp.replace("|", "\\|")
        vi = ln.vi.replace("|", "\\|")
        out.append(f"| {ln.seq} | {ln.ts} | {jp} | {vi} |")
    return "\n".join(out) + "\n"


def to_json(lines: list[TranscriptLine]) -> str:
    """Render lines as a JSON array."""
    return json.dumps(
        [
            {
                "seq": ln.seq,
                "ts": ln.ts,
                "start_ms": round(ln.start_ms, 1),
                "end_ms": round(ln.end_ms, 1),
                "jp": ln.jp,
                "vi": ln.vi,
            }
            for ln in lines
        ],
        ensure_ascii=False,
        indent=2,
    )


_RENDERERS = {
    "txt": to_text,
    "text": to_text,
    "md": to_markdown,
    "markdown": to_markdown,
    "srt": to_srt,
    "json": to_json,
}


def render(lines: list[TranscriptLine], fmt: str) -> str:
    """Render lines in the requested format (txt/md/srt/json)."""
    try:
        return _RENDERERS[fmt](lines)
    except KeyError as exc:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown format: {fmt}") from exc


def export_file(log_path: str | Path, fmt: str, out_path: Optional[str | Path]) -> str:
    """Load ``log_path``, render as ``fmt``, optionally write to ``out_path``."""
    lines = build_lines(load_events(log_path))
    text = render(lines, fmt)
    if out_path is not None:
        Path(out_path).write_text(text, encoding="utf-8")
    return text


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a bilingual transcript from a JSONL evidence log."
    )
    parser.add_argument("log", help="path to the JSONL evidence log")
    parser.add_argument(
        "--format",
        "-f",
        default="txt",
        choices=sorted(_RENDERERS),
        help="output format (default: txt)",
    )
    parser.add_argument(
        "--out", "-o", default=None, help="write to this file instead of stdout"
    )
    parser.add_argument(
        "--stats", action="store_true", help="also print run summary stats to stderr"
    )
    args = parser.parse_args(argv)

    events = load_events(args.log)
    lines = build_lines(events)
    text = render(lines, args.format)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(lines)} lines to {args.out}", file=sys.stderr)
    else:
        print(text)
    if args.stats:
        print(json.dumps(summarize(events), ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
