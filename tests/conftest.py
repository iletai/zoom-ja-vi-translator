"""Shared test fixtures for the Zoom JA→VI translator test suite."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from src.llm_translator import LlmTranslator


def build_llm_translator(llm: Any, *, keep_context: bool = False) -> LlmTranslator:
    """Build an LlmTranslator for unit tests without loading a real GGUF model.

    Bypasses ``__init__`` (which loads llama.cpp) via ``__new__`` and sets exactly
    the attributes the translation/post-processing paths read. Keep this in sync
    with ``LlmTranslator.__init__`` — the whole point of centralizing it here is
    that a new attribute added to __init__ is set in ONE place, not copy-pasted
    across every test module (which silently broke tests twice before).

    ``llm`` is the caller's mock (each test module has its own _DummyLlm variant).
    """
    t = LlmTranslator.__new__(LlmTranslator)
    t._keep_context = keep_context
    t._history = deque(maxlen=3 if keep_context else 1)
    t._lock = threading.Lock()
    t._chinese_logit_bias = {}
    t._vi_grammar = None
    t._fast_translator = None
    t.system_prompt = "test"
    t.context_sentences = 3 if keep_context else 0
    t.temperature = 0.1
    t.top_p = 0.3
    t.frequency_penalty = 0.1
    t.max_tokens = 150
    t.n_ctx = 768
    # Longest-first lookup tables — built in __init__ from the class-level maps.
    t._sorted_jp_dow = sorted(LlmTranslator._JP_DOW_MAP.items(), key=lambda x: -len(x[0]))
    t._sorted_katakana_terms = sorted(
        LlmTranslator._KATAKANA_TERM_MAP.items(), key=lambda x: -len(x[0]))
    t._sorted_proper_nouns = sorted(
        LlmTranslator._PROPER_NOUN_MAP.items(), key=lambda x: -len(x[0]))
    t.llm = llm
    return t
