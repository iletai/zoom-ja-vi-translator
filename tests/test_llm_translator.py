from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_translator import LlmTranslator  # noqa: E402


class _DummyLlm:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def create_chat_completion(self, *, messages, **_: object) -> dict[str, object]:
        self.messages = messages
        return {"choices": [{"message": {"content": "Bản dịch thử"}}]}


def _make_translator() -> tuple[LlmTranslator, _DummyLlm]:
    translator = LlmTranslator.__new__(LlmTranslator)
    translator._keep_context = False
    translator._history = deque(maxlen=1)
    translator._lock = threading.Lock()
    translator.system_prompt = "test"
    translator.context_sentences = 0
    translator.temperature = 0.1
    translator.top_p = 0.3
    translator.frequency_penalty = 0.1
    translator.max_tokens = 150
    dummy_llm = _DummyLlm()
    translator.llm = dummy_llm
    return translator, dummy_llm


def test_clean_translation_rejects_meta_greeting_and_promise() -> None:
    text = "Xin chào, bạn! Tôi sẽ dịch đoạn văn này sang tiếng Việt:"
    assert LlmTranslator._clean_translation(text) == ""


def test_clean_translation_salvages_text_after_meta_prefix() -> None:
    text = "Đây là bản dịch sang tiếng Việt: Chào mừng đến thế giới mơ ước"
    assert LlmTranslator._clean_translation(text) == "Chào mừng đến thế giới mơ ước"


def test_clean_translation_keeps_non_meta_colons() -> None:
    text = "Lúc 10:30 hệ thống bắt đầu"
    assert LlmTranslator._clean_translation(text) == "Lúc 10:30 hệ thống bắt đầu"


def test_clean_translation_rejects_short_english_output() -> None:
    assert LlmTranslator._clean_translation("Hello everyone.") == ""
    assert LlmTranslator._clean_translation("Yes, it's difficult.") == ""


def test_clean_translation_keeps_single_it_term() -> None:
    assert LlmTranslator._clean_translation("Cloud") == "Cloud"
    assert LlmTranslator._clean_translation("API") == "API"


def test_clean_translation_keeps_short_unaccented_vietnamese() -> None:
    assert LlmTranslator._clean_translation("cho con") == "cho con"


def test_translate_one_replaces_proper_nouns_after_katakana_preprocessing() -> None:
    translator, dummy_llm = _make_translator()

    result = translator._translate_one("テクノロジー社は秋葉原に本社があり", update_context=False)

    assert result == "Bản dịch thử"
    assert dummy_llm.messages is not None
    assert dummy_llm.messages[-2] == {
        "role": "user",
        "content": "JA: Technology社はAkihabaraに本社があり",
    }
    assert dummy_llm.messages[-1] == {"role": "assistant", "content": "VI:"}