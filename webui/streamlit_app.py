"""Streamlit live dashboard for the Zoom JA→VI translator.

Decoupled by design: the translator process writes a JSONL evidence log
(`--log PATH` or `ZT_EVIDENCE_LOG=PATH`); this app *tails* that log and renders a
live bilingual subtitle feed plus run metrics. It never touches the audio
pipeline, so it cannot add latency or drop data — and it keeps working even if
restarted mid-meeting.

Run:
    streamlit run webui/streamlit_app.py
Then start the translator with logging, e.g.:
    python main.py --system-audio --log test_audio/evidence/live.jsonl
and pick that file in the sidebar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable regardless of where streamlit is launched from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.transcript_export import (  # noqa: E402
    build_lines,
    load_events,
    render,
    summarize,
)

EVIDENCE_DIR = _ROOT / "test_audio" / "evidence"

st.set_page_config(
    page_title="Zoom JA→VI Translator — Live",
    page_icon="🎙️",
    layout="wide",
)


def _discover_logs() -> list[Path]:
    """Return evidence logs, newest first."""
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(EVIDENCE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _sidebar() -> tuple[Path | None, int, int, bool]:
    st.sidebar.title("🎙️ Live Translator")
    logs = _discover_logs()
    options = {p.name: p for p in logs}

    manual = st.sidebar.text_input(
        "Evidence log path",
        value="",
        placeholder="leave blank to pick below",
        help="Absolute or project-relative path to a JSONL evidence log.",
    )
    chosen: Path | None = None
    if manual.strip():
        p = Path(manual.strip())
        chosen = p if p.is_absolute() else _ROOT / p
    elif options:
        name = st.sidebar.selectbox("…or pick a recent log", list(options))
        chosen = options[name]
    else:
        st.sidebar.info("No logs in test_audio/evidence/. Run the translator with --log.")

    st.sidebar.divider()
    auto = st.sidebar.toggle("Auto-refresh (live tail)", value=True)
    interval = st.sidebar.slider("Refresh every (s)", 1, 10, 2, disabled=not auto)
    tail = st.sidebar.slider("Show last N lines", 10, 500, 80, step=10)
    return chosen, interval, tail, auto


def _render_metrics(stats: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", stats["segments"])
    c2.metric("Max-cut %", f'{stats["max_utterance_pct"]}%')
    c3.metric("Median latency", f'{stats["translate_latency_median_ms"]:.0f} ms')
    loss_total = sum(stats["loss_events"].values())
    c4.metric("Loss events", loss_total, delta=None)

    with st.expander("Run details", expanded=False):
        st.write("Segment end reasons:", stats["reasons"])
        st.write(
            f'Median segment {stats["duration_median_s"]}s · '
            f'max {stats["duration_max_s"]}s · '
            f'max latency {stats["translate_latency_max_ms"]:.0f} ms'
        )
        if stats["loss_events"]:
            st.warning(f'Loss/anomaly events: {stats["loss_events"]}')
        else:
            st.success("No loss/anomaly events recorded ✓")


def _render_feed(lines, tail: int) -> None:
    if not lines:
        st.info("Waiting for translated lines…")
        return
    shown = lines[-tail:]
    for ln in shown:
        ts = ln.ts or ""
        st.markdown(
            f"<div style='margin-bottom:0.6rem'>"
            f"<span style='color:#888;font-size:0.8rem'>[{ts}] #{ln.seq}</span><br>"
            f"<span style='color:#22a7f0'>🇯🇵 {ln.jp}</span><br>"
            f"<span style='color:#2ecc71;font-weight:600'>🇻🇳 {ln.vi}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_downloads(events, lines) -> None:
    if not lines:
        return
    st.sidebar.divider()
    st.sidebar.subheader("💾 Export transcript")
    stem = "transcript"
    st.sidebar.download_button(
        "Text (.txt)", render(lines, "txt"), f"{stem}.txt", "text/plain"
    )
    st.sidebar.download_button(
        "Subtitles (.srt)", render(lines, "srt"), f"{stem}.srt", "text/plain"
    )
    st.sidebar.download_button(
        "Markdown (.md)", render(lines, "md"), f"{stem}.md", "text/markdown"
    )
    st.sidebar.download_button(
        "JSON (.json)", render(lines, "json"), f"{stem}.json", "application/json"
    )


def main() -> None:
    chosen, interval, tail, auto = _sidebar()

    st.title("Bilingual Live Subtitles — JA → VI")
    if chosen is None:
        st.stop()
    if not chosen.exists():
        st.error(f"Log not found: {chosen}")
        st.stop()
    st.caption(f"Tailing `{chosen}`")

    # st.fragment(run_every=...) reruns only this fragment on a timer, so the live
    # feed refreshes without re-running the whole script (Streamlit ≥1.33).
    @st.fragment(run_every=interval if auto else None)
    def live_view() -> None:
        events = load_events(chosen)
        lines = build_lines(events)
        stats = summarize(events)
        _render_metrics(stats)
        st.divider()
        _render_feed(lines, tail)
        # Stash for the download buttons rendered in the sidebar (outside fragment).
        st.session_state["_events"] = events
        st.session_state["_lines"] = lines

    live_view()
    _render_downloads(
        st.session_state.get("_events", []), st.session_state.get("_lines", [])
    )


if __name__ == "__main__":
    main()
