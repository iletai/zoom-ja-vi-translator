from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_translator import LlmTranslator  # noqa: E402


class _DummyLlm:
    def __init__(self, response_text: str = "Bản dịch thử") -> None:
        self.messages: list[dict[str, str]] | None = None
        self.prompt: str | None = None
        self.response_text = response_text

    def create_chat_completion(self, *, messages, **_: object) -> dict[str, object]:
        self.messages = messages
        return {"choices": [{"message": {"content": self.response_text}}]}

    def create_completion(self, *, prompt, **_: object) -> dict[str, object]:
        self.prompt = prompt
        return {"choices": [{"text": self.response_text}]}


def _make_translator(response_text: str = "Bản dịch thử") -> tuple[LlmTranslator, _DummyLlm]:
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
    translator.n_ctx = 768
    dummy_llm = _DummyLlm(response_text=response_text)
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


def test_translate_one_uses_added_filler_phrase_overrides() -> None:
    translator, _ = _make_translator()

    expected = {
        "いえいえこちらこそ": "Không không, bên tôi mới phải cảm ơn",
        "先日は打ち合わせありがとうございました": "Cảm ơn về cuộc họp hôm trước",
        "先日はありがとうございました": "Cảm ơn về hôm trước",
        "いかがですか": "Thế nào ạ?",
        "いかがでしょうか": "Thế nào ạ?",
        "難しいですか": "Có khó không?",
        "難しいと思います": "Tôi nghĩ là khó",
    }

    for source, translation in expected.items():
        assert translator._translate_one(source, update_context=False) == translation


def test_translate_one_matches_truncated_filler_prefix() -> None:
    translator, _ = _make_translator()

    assert translator._translate_one("ありがとうございま", update_context=False) == "Cảm ơn"


def test_clean_translation_keeps_short_unaccented_vietnamese() -> None:
    assert LlmTranslator._clean_translation("cho con") == "cho con"


def test_translate_one_replaces_proper_nouns_after_katakana_preprocessing() -> None:
    translator, dummy_llm = _make_translator()

    result = translator._translate_one("テクノロジー社は秋葉原に本社があり", update_context=False)

    assert result == "Bản dịch thử"
    assert dummy_llm.prompt is not None
    assert "<|im_start|>user\nJA: Technology社はAkihabaraに本社があり<|im_end|>\n" in dummy_llm.prompt
    assert dummy_llm.prompt.endswith("<|im_start|>assistant\nVI: ")


def test_translate_one_uses_added_katakana_term_overrides() -> None:
    translator, _ = _make_translator()

    assert translator._translate_one("トークイベント", update_context=False) == "talk event"
    assert translator._translate_one("トークイーブメント", update_context=False) == "talk event"


def test_build_raw_prompt_uses_true_prefill() -> None:
    translator, _ = _make_translator()

    prompt = translator._build_raw_prompt("会議を始めます")

    assert prompt.endswith("<|im_start|>assistant\nVI: ")
    assert not prompt.endswith("<|im_start|>assistant\nVI: <|im_end|>")


def test_translate_one_strips_uoc_bias_prefix_for_meeting_sentence() -> None:
    translator, _ = _make_translator("Ước mong không tiến triển")

    result = translator._translate_one(
        "前回からそんなたってないんで進捗はないと思いますけどお願いします",
        update_context=False,
    )

    assert result == "Không tiến triển"


def test_translate_one_rewrites_uoc_bias_to_hi_vong_for_hope_source() -> None:
    translator, _ = _make_translator("Ước mơ ở mức độ này")

    result = translator._translate_one("あくまで希望のレベルですね", update_context=False)

    assert result == "Hi vọng ở mức độ này"


def test_translate_one_keeps_literal_uoc_for_dream_source() -> None:
    translator, _ = _make_translator("Ước mơ trong tương lai")

    result = translator._translate_one("将来の夢について話します", update_context=False)

    assert result == "Ước mơ trong tương lai"
