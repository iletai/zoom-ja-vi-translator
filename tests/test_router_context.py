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


def test_filler_map_shared_with_llm_backend() -> None:
    """Router and LLM must reference the SAME FILLER_MAP so they never drift.

    Two separate dicts drifted to 22 vs 72 entries before consolidation; this
    guards that both backends stay pinned to src.domain_data.FILLER_MAP.
    """
    from src.domain_data import FILLER_MAP as shared
    assert _FILLER_MAP is shared, "router _FILLER_MAP must be the shared FILLER_MAP"
    # A meeting greeting only the LLM used to have — now the router bypasses too.
    assert _FILLER_MAP.get("お疲れ様です") == "Xin chào"
    assert _FILLER_MAP.get("ありがとうございます") == "Cảm ơn"


def test_match_filler_rejects_real_words_that_prefix_a_filler() -> None:
    """Short real words must NOT prefix-match a longer filler (mistranslation).

    です (copula "is"), なる (verb "become"), そう (adverb) each prefix a filler
    (ですね/なるほど/そうそう) but are real content. The >=3-char floor + 1-mora
    truncation window keep them out; a valid 1-mora-cut filler still matches.
    """
    from src.domain_data import match_filler
    for real_word in ("です", "なる", "そう", "は", "が"):
        assert match_filler(real_word) is None, f"{real_word!r} must reach real translation"
    # Genuine ASR truncation (final mora dropped) still short-circuits.
    assert match_filler("うんうんう") == "Vâng vâng vâng"
    assert match_filler("ありがとうございま") == "Cảm ơn"


def test_filler_not_added_to_history() -> None:
    """Filler short-circuits must NOT enter the context window.

    A はい→"Vâng" pair anchors no terminology; three back-channels in a row would
    evict every real turn from the maxlen=3 window. Fillers skip history; real
    sentences still enter it.
    """
    t = _make_translator(context_sentences=3)
    t._post_with_retry = lambda _m: "Câu thật."
    for f in ("はい", "うん", "ええ"):
        t._translate_one(f)
    assert len(t._history) == 0, "fillers must not populate history"
    t._translate_one("本当の文です")
    assert len(t._history) == 1, "a real sentence must still enter history"


def test_filler_not_added_to_history_via_translate_many() -> None:
    """The batch path (translate_many) must also exclude fillers from history.

    _translate_one skips fillers, but translate_many appends batch results in a
    separate post-loop — it must apply the SAME exclusion or fillers leak back in
    (the pipeline's hot path is translate_many, so this is the real risk).
    """
    t = _make_translator(context_sentences=3)
    t._post_with_retry = lambda m: "Vâng" if "はい" in m[-1]["content"] else "Câu thật."
    t.translate_many(["これはテストです", "はい", "終わりました"])
    keys = [src for src, _ in t._history]
    assert "はい" not in keys, "filler leaked into history via translate_many"
    assert keys == ["これはテストです", "終わりました"], "real turns must remain, in order"


def test_hard_refusal_rejected_and_not_in_history() -> None:
    """A model refusal must be rejected AND kept out of the context window.

    Router used to only catch prefix refusals; a mid-string 'I cannot translate'
    slipped through and poisoned the next 3 segments' context.
    """
    t = _make_translator(context_sentences=3)
    t._post_with_retry = lambda _m: "I cannot translate that."
    result = t._translate_one("これはテストです")
    assert result == "", "hard refusal must be rejected"
    assert len(t._history) == 0, "refusal must not poison history"
    # A valid translation containing a soft phrase is NOT rejected.
    t._post_with_retry = lambda _m: "Tôi hiểu rồi, cảm ơn anh."
    assert t._translate_one("分かりましたありがとう") == "Tôi hiểu rồi, cảm ơn anh."


def test_long_valid_translation_not_rejected() -> None:
    """A long ascii-heavy VI sentence (romanized names + acronyms) must pass.

    A previous length>400 guard false-rejected these — the 180-token cap already
    bounds runaways, so no char-length ceiling is needed. This guards against it
    being re-added.
    """
    from src.router_translator import _looks_like_refusal_or_echo
    long_vi = "Fukase-san báo cáo DMAT rằng bệnh nhân tại CROSS-TENANT " * 9  # ~490 chars
    assert not _looks_like_refusal_or_echo("元の文", long_vi), "long valid VI must not be rejected"


def test_http_retry_only_on_retryable_status() -> None:
    """400/401 fail fast (1 attempt); 429/5xx retry (2 attempts).

    raise_for_status() raises HTTPError (a RequestException subclass); without
    status classification, every 4xx wasted a second round-trip on the hot path.
    """
    import requests
    import unittest.mock as _mock
    t = _make_translator()
    for code, want_attempts in ((400, 1), (401, 1), (429, 2), (500, 2), (503, 2)):
        attempts = [0]

        def post(*_a, code=code, **_k):
            attempts[0] += 1
            resp = _mock.Mock()
            resp.status_code = code
            resp.text = ""
            resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
            return resp

        t._session.post = post  # type: ignore[assignment]
        t._post_with_retry([{"role": "user", "content": "x"}])
        assert attempts[0] == want_attempts, f"HTTP {code}: {attempts[0]} attempts, want {want_attempts}"


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


# ── System-prompt structural invariants ────────────────────────────────────
# The router prompt is load-bearing config refined over many iterations. These
# assert the invariants stay intact through refactors — none call the model, so
# they are fast and deterministic. A silent revert (which git-staging hazards
# caused before) is caught here instead of only in a live meeting.

def test_router_prompt_has_xml_section_structure() -> None:
    import config
    p = config._ROUTER_DEFAULT_PROMPT
    assert "<rules>" in p and "</rules>" in p, "rules section lost"
    assert "<examples>" in p and "</examples>" in p, "examples section lost"
    # Sandwich defense must be the LAST instruction (recency anchor).
    assert p.rstrip().endswith("next message."), "sandwich no longer last"
    # Order: rules before examples before the Remember sandwich.
    assert p.index("<rules>") < p.index("<examples>") < p.index("Remember:")


def test_router_prompt_functional_word_rule_not_reverted() -> None:
    """The content-fragment fix (verified live) must survive refactors.

    ちょっと was removed from the 'content-free' list because it HAS meaning;
    leaving it there made the model collapse content fragments like
    ちょっとその件については to a bare '...'. Guard against a silent revert.
    """
    import config
    p = config._ROUTER_DEFAULT_PROMPT
    assert "content-free particle" in p, "functional-word fix reverted"
    assert "single functional word" not in p, "old (buggy) wording is back"
    # ちょっと must NOT be listed as a content-free particle.
    particle_rule = p[p.index("content-free particle"):p.index("content-free particle") + 120]
    assert "ちょっと" not in particle_rule, "ちょっと wrongly listed as content-free again"


def test_router_prompt_keeps_core_rules() -> None:
    import config
    p = config._ROUTER_DEFAULT_PROMPT
    for marker in ("ませんか", "自分", "Prior turns", "romaji", "one line"):
        assert marker in p, f"core rule marker {marker!r} missing from prompt"
