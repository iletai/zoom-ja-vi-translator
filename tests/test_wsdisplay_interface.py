"""WsDisplay must satisfy the SubtitleDisplay interface the pipeline calls.

WsDisplay (web-UI backend) is a drop-in replacement for SubtitleDisplay
(terminal). The pipeline calls display methods by KEYWORD (e.g.
``show_source(japanese, seq=seq)``), so a parameter-name drift between the two
crashes the ASR worker at runtime — exactly what happened when show_source's
``seq`` was renamed to ``_seq`` (TypeError: unexpected keyword argument 'seq').

These tests compare the two classes' method signatures so that regression is
caught in CI instead of only in a live meeting. No gateway / async loop needed.
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.display import SubtitleDisplay  # noqa: E402
from webui.host_server import WsDisplay  # noqa: E402

# Methods the pipeline drives on whatever display is wired in. Each must exist on
# WsDisplay with parameter NAMES matching SubtitleDisplay, because the pipeline
# passes some of them by keyword.
_PIPELINE_DISPLAY_METHODS = (
    "show_source",
    "show_target",
    "show_pair",
    "show_source_partial",
    "finalize_source",
    "info",
)


def test_wsdisplay_has_every_pipeline_display_method() -> None:
    for name in _PIPELINE_DISPLAY_METHODS:
        assert hasattr(WsDisplay, name), f"WsDisplay missing {name}() the pipeline calls"


def test_wsdisplay_param_names_match_subtitledisplay() -> None:
    """Parameter names must match — the pipeline calls several methods by keyword."""
    for name in _PIPELINE_DISPLAY_METHODS:
        ref = inspect.signature(getattr(SubtitleDisplay, name))
        got = inspect.signature(getattr(WsDisplay, name))
        ref_params = [p for p in ref.parameters if p != "self"]
        got_params = [p for p in got.parameters if p != "self"]
        assert got_params == ref_params, (
            f"WsDisplay.{name}{tuple(got_params)} params differ from "
            f"SubtitleDisplay.{name}{tuple(ref_params)} — a keyword call from the "
            f"pipeline (e.g. seq=...) will raise TypeError at runtime"
        )


def test_wsdisplay_accepts_the_exact_keyword_calls_pipeline_makes() -> None:
    """Reproduce the pipeline's real call sites against a stub connection."""
    import asyncio

    class _FakeConn:
        async def send(self, obj: dict) -> None:  # coroutine, like Connection.send
            pass

    loop = asyncio.new_event_loop()
    try:
        d = WsDisplay(_FakeConn(), loop)
        # These mirror pipeline.py:408/428/616/712/739 exactly.
        d.show_source("会議です", seq=5)
        d.show_target("Cuộc họp", japanese="会議です", seq=5)
        d.show_source("会議です")
        d.show_pair("会議です", "Cuộc họp")
        d.finalize_source("会議です")
        d.show_source_partial("会議", "です")
        d.info("status")
    finally:
        loop.close()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print(f"PASS {_name}")
    print("=== all WsDisplay interface tests passed ===")
