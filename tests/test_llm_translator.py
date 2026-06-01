from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_translator import LlmTranslator  # noqa: E402


def test_clean_translation_rejects_meta_greeting_and_promise() -> None:
    text = "Xin chào, bạn! Tôi sẽ dịch đoạn văn này sang tiếng Việt:"
    assert LlmTranslator._clean_translation(text) == ""


def test_clean_translation_salvages_text_after_meta_prefix() -> None:
    text = "Đây là bản dịch sang tiếng Việt: Chào mừng đến thế giới mơ ước"
    assert LlmTranslator._clean_translation(text) == "Chào mừng đến thế giới mơ ước"


def test_clean_translation_keeps_non_meta_colons() -> None:
    text = "Lúc 10:30 hệ thống bắt đầu"
    assert LlmTranslator._clean_translation(text) == "Lúc 10:30 hệ thống bắt đầu"
