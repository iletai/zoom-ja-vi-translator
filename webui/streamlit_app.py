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
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Make `src`/`webui` importable regardless of where streamlit is launched from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.transcript_export import (  # noqa: E402
    build_lines,
    load_events,
    render,
    summarize,
)
from webui.filters import latency_alert, prepare_feed, safe_stem  # noqa: E402

EVIDENCE_DIR = _ROOT / "test_audio" / "evidence"

st.set_page_config(
    page_title="Zoom JA→VI Translator — Live",
    page_icon="🎙️",
    layout="wide",
)


@dataclass
class Settings:
    """All user-controllable dashboard options, gathered from the sidebar."""

    log: Path | None
    auto: bool
    interval: int
    tail: int
    query: str
    show_jp: bool
    show_vi: bool
    newest_first: bool
    font_px: int
    latency_threshold_ms: int
    export_stem: str


def _discover_logs() -> list[Path]:
    """Return evidence logs, newest first."""
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(EVIDENCE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _pick_log() -> Path | None:
    logs = _discover_logs()
    options = {p.name: p for p in logs}
    manual = st.sidebar.text_input(
        "Evidence log path",
        value="",
        placeholder="leave blank to pick below",
        help="Absolute or project-relative path to a JSONL evidence log.",
    )
    if manual.strip():
        p = Path(manual.strip())
        return p if p.is_absolute() else _ROOT / p
    if options:
        name = st.sidebar.selectbox("…or pick a recent log", list(options))
        return options[name]
    st.sidebar.info("No logs in test_audio/evidence/. Run the translator with --log.")
    return None


def _sidebar() -> Settings:
    st.sidebar.title("🎙️ Live Translator")
    log = _pick_log()

    st.sidebar.divider()
    st.sidebar.subheader("🔄 Live tail")
    auto = st.sidebar.toggle("Auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh every (s)", 1, 10, 2, disabled=not auto)
    tail = st.sidebar.slider("Show last N lines", 10, 500, 80, step=10)

    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Display")
    query = st.sidebar.text_input(
        "Filter (search JP/VI)", value="", placeholder="type to filter lines…"
    )
    lang = st.sidebar.radio(
        "Languages", ["Both", "VI only", "JP only"], horizontal=True
    )
    newest_first = st.sidebar.toggle("Newest first", value=False)
    font_px = st.sidebar.slider("Font size (px)", 12, 28, 16)
    latency_threshold_ms = st.sidebar.number_input(
        "Latency alert above (ms, 0=off)", min_value=0, max_value=20000, value=4000, step=500
    )

    st.sidebar.divider()
    st.sidebar.subheader("💾 Export")
    export_stem = st.sidebar.text_input("Filename", value="transcript")

    return Settings(
        log=log,
        auto=auto,
        interval=interval,
        tail=tail,
        query=query,
        show_jp=lang in ("Both", "JP only"),
        show_vi=lang in ("Both", "VI only"),
        newest_first=newest_first,
        font_px=font_px,
        latency_threshold_ms=int(latency_threshold_ms),
        export_stem=export_stem,
    )


def _render_metrics(stats: dict, latency_threshold_ms: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", stats["segments"])
    c2.metric("Max-cut %", f'{stats["max_utterance_pct"]}%')
    median = stats["translate_latency_median_ms"]
    over = latency_alert(median, latency_threshold_ms)
    c3.metric(
        "Median latency",
        f"{median:.0f} ms",
        delta="⚠ high" if over else None,
        delta_color="inverse" if over else "normal",
    )
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


def _render_feed(lines, cfg: Settings) -> None:
    shown = prepare_feed(lines, cfg.query, cfg.tail, cfg.newest_first)
    if not shown:
        if cfg.query.strip():
            st.info(f"No lines match “{cfg.query.strip()}”.")
        else:
            st.info("Waiting for translated lines…")
        return
    st.caption(f"Showing {len(shown)} line(s)")
    meta_px = max(10, cfg.font_px - 4)
    for ln in shown:
        ts = ln.ts or ""
        parts = [
            f"<div style='margin-bottom:0.6rem;font-size:{cfg.font_px}px'>",
            f"<span style='color:#888;font-size:{meta_px}px'>[{ts}] #{ln.seq}</span>",
        ]
        if cfg.show_jp:
            parts.append(f"<br><span style='color:#22a7f0'>🇯🇵 {ln.jp}</span>")
        if cfg.show_vi:
            parts.append(
                f"<br><span style='color:#2ecc71;font-weight:600'>🇻🇳 {ln.vi}</span>"
            )
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)


def _render_downloads(lines, stem: str) -> None:
    if not lines:
        return
    safe = safe_stem(stem)
    st.sidebar.download_button(
        "Text (.txt)", render(lines, "txt"), f"{safe}.txt", "text/plain"
    )
    st.sidebar.download_button(
        "Subtitles (.srt)", render(lines, "srt"), f"{safe}.srt", "text/plain"
    )
    st.sidebar.download_button(
        "Markdown (.md)", render(lines, "md"), f"{safe}.md", "text/markdown"
    )
    st.sidebar.download_button(
        "JSON (.json)", render(lines, "json"), f"{safe}.json", "application/json"
    )


def main() -> None:
    cfg = _sidebar()

    st.title("Bilingual Live Subtitles — JA → VI")
    if cfg.log is None:
        st.stop()
    if not cfg.log.exists():
        st.error(f"Log not found: {cfg.log}")
        st.stop()
    st.caption(f"Tailing `{cfg.log}`")

    # st.fragment(run_every=...) reruns only this fragment on a timer, so the live
    # feed refreshes without re-running the whole script (Streamlit ≥1.33).
    @st.fragment(run_every=cfg.interval if cfg.auto else None)
    def live_view() -> None:
        events = load_events(cfg.log)
        lines = build_lines(events)
        stats = summarize(events)
        _render_metrics(stats, cfg.latency_threshold_ms)
        st.divider()
        _render_feed(lines, cfg)
        # Stash the full (unfiltered) transcript so downloads always export the
        # complete run, regardless of the on-screen filter/tail.
        st.session_state["_lines"] = lines

    live_view()
    _render_downloads(st.session_state.get("_lines", []), cfg.export_stem)


if __name__ == "__main__":
    main()
