"""Tests for RouterTranslator — filler short-circuit, _preprocess, _build_messages,
and sequential vs parallel multi-sentence routing. No real HTTP calls."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import unittest.mock as mock

# Patch requests.Session so RouterTranslator.__init__ doesn't blow up
# (it calls warmup() which calls _post_with_retry via _translate_one).
with mock.patch("requests.Session"):
    from src.router_translator import RouterTranslator, _FILLER_MAP


def _make_translator(context_sentences: int = 2) -> RouterTranslator:
    """Build a RouterTranslator with a mocked HTTP session; no real gateway needed."""
    with mock.patch("requests.Session"), mock.patch.object(
        RouterTranslator, "warmup"  # suppress warmup so no HTTP on __init__
    ):
        os.environ.setdefault("ZT_ROUTER_KEY", "test-key")
        t = RouterTranslator.__new__(RouterTranslator)
        from collections import deque
        import threading

        t.base_url = "http://127.0.0.1:20128/v1"
        t.url = f"{t.base_url}/chat/completions"
        t.model = "test-model"
        t.api_key = "sk-test"
        t.system_prompt = "Translate JA→VI."
        t.temperature = 0.0
        t.max_tokens = 180
        t.timeout = 12.0
        t.context_sentences = max(0, context_sentences)
        t.max_parallel = 4
        t._keep_context = context_sentences > 0
        t._history = deque(maxlen=context_sentences if context_sentences > 0 else 1)
        t._lock = threading.Lock()
        t._session = mock.MagicMock()
    return t


# ── 1. _FILLER_MAP lookups ──────────────────────────────────────────────────

def test_filler_hai() -> None:
    assert _FILLER_MAP.get("はい") == "Vâng"


def test_filler_socca() -> None:
    assert _FILLER_MAP.get("そっか") == "Vậy à"


def test_filler_short_circuits(monkeypatch: "pytest.MonkeyPatch") -> None:
    """_translate_one must return filler result without calling _post_with_retry."""
    t = _make_translator()
    called = []
    monkeypatch.setattr(t, "_post_with_retry", lambda msgs: called.append(msgs) or "X")
    result = t._translate_one("はい")
    assert result == "Vâng"
    assert called == [], "HTTP should not be called for filler input"


# ── 2. _preprocess ──────────────────────────────────────────────────────────

def test_preprocess_surname_san() -> None:
    result = RouterTranslator._preprocess("深瀬さんに連絡します")
    assert "Fukase-san" in result, f"Expected Fukase-san in {result!r}"


def test_preprocess_katakana_loanword() -> None:
    result = RouterTranslator._preprocess("フェーズ2に入ります")
    assert "phase" in result.lower(), f"Expected 'phase' in {result!r}"


# ── 3. _build_messages ──────────────────────────────────────────────────────

def test_build_messages_no_history() -> None:
    t = _make_translator(context_sentences=2)
    msgs = t._build_messages("テスト")
    # [system, user(current)] — 2 messages, no history
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "テスト" in msgs[1]["content"]


def test_build_messages_with_history() -> None:
    t = _make_translator(context_sentences=2)
    # Inject one history entry (raw_cleaned, translation)
    t._history.append(("前の文", "Câu trước"))

    msgs = t._build_messages("現在の文")
    # [system, user(hist0), assistant(hist0), user(current)] — 4 messages
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"      # history source
    assert msgs[2]["role"] == "assistant" # history translation
    assert msgs[3]["role"] == "user"      # current
    assert "Câu trước" in msgs[2]["content"]
    assert "現在の文" in msgs[3]["content"]


def test_build_messages_history_preprocessed() -> None:
    """History source turns must be preprocessed (romanized) in the messages."""
    t = _make_translator(context_sentences=2)
    # Raw history contains a kanji surname
    t._history.append(("深瀬さんが言いました", "Fukase-san đã nói"))
    msgs = t._build_messages("次")
    # history user turn should have Fukase-san, not the original kanji
    history_user_content = msgs[1]["content"]
    assert "Fukase-san" in history_user_content, (
        f"History user turn not preprocessed: {history_user_content!r}"
    )


def test_build_messages_context_off_no_history_injected() -> None:
    """With context_sentences=0, history is never appended to messages."""
    t = _make_translator(context_sentences=0)
    t._history.append(("前の文", "Câu trước"))  # manually stuffed
    msgs = t._build_messages("テスト")
    assert len(msgs) == 2  # only [system, user(current)]


# ── 4. Multi-sentence with context ON → sequential ─────────────────────────

def test_multi_sentence_context_on_sequential(monkeypatch: "pytest.MonkeyPatch") -> None:
    """With context ON, sentences are translated one-by-one so each enters history."""
    t = _make_translator(context_sentences=2)
    call_count = []

    def fake_post(_msgs: list) -> str:
        call_count.append(len(t._history))  # snapshot history depth at call time
        return "Kết quả"

    monkeypatch.setattr(t, "_post_with_retry", fake_post)

    # Two sentences; split_japanese_sentences must produce ≥2 for this path to fire.
    # Force it by patching.
    import src.router_translator as _rt
    monkeypatch.setattr(_rt, "split_japanese_sentences", lambda _: ["文A。", "文B。"])

    import config as _cfg
    monkeypatch.setattr(_cfg, "TRANSLATE_SPLIT_SENTENCES", True)

    t.translate("文A。文B。")

    # Both sentences should have been translated (2 HTTP calls) and history grows.
    assert len(call_count) == 2
    # After both translated, history should have 2 entries (both entered history).
    assert len(t._history) == 2


# ── 5. Multi-sentence with context OFF → parallel ──────────────────────────

def test_multi_sentence_context_off_parallel(monkeypatch: "pytest.MonkeyPatch") -> None:
    """With context OFF, all sentences are translated in parallel; history stays empty."""
    t = _make_translator(context_sentences=0)
    call_count = []

    def fake_post(_msgs: list) -> str:
        call_count.append(True)
        return "Kết quả"

    monkeypatch.setattr(t, "_post_with_retry", fake_post)

    import src.router_translator as _rt
    monkeypatch.setattr(_rt, "split_japanese_sentences", lambda _: ["文A。", "文B。"])

    import config as _cfg
    monkeypatch.setattr(_cfg, "TRANSLATE_SPLIT_SENTENCES", True)

    t.translate("文A。文B。")

    assert len(call_count) == 2
    # Context is off — history must stay empty
    assert len(t._history) == 0
