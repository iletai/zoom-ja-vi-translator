"""Tests for the improved _clean_translation refusal/preamble handling."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_translator import LlmTranslator  # noqa: E402


def test_preamble_colon_no_content_returns_empty():
    """Preamble ending with colon but nothing after -> empty (triggers retry)."""
    assert LlmTranslator._clean_translation("Tôi sẽ dịch đoạn văn như sau:") == ""


def test_hard_refusal_returns_empty():
    """Hard refusal patterns should reject."""
    text = "Tôi xin lỗi, nhưng tôi không thể dịch đoạn văn này"
    assert LlmTranslator._clean_translation(text) == ""


def test_preamble_plus_content_extracts_content():
    """Preamble + colon + valid content should extract the content."""
    text = "Đây là bản dịch sang tiếng Việt: Có cái nhỏ có cái lớn"
    assert LlmTranslator._clean_translation(text) == "Có cái nhỏ có cái lớn"


def test_multiline_skips_preamble_line():
    """Multi-line output: skip preamble lines, use first valid line."""
    text = "Tôi sẽ dịch đoạn văn:\nCó cái nhỏ có cái lớn vậy đó"
    assert LlmTranslator._clean_translation(text) == "Có cái nhỏ có cái lớn vậy đó"


def test_valid_translation_passes_through():
    """Normal Vietnamese translation passes through unchanged."""
    text = "Có sai lệch nhưng cũng được nhỉ"
    assert LlmTranslator._clean_translation(text) == "Có sai lệch nhưng cũng được nhỉ"


def test_leading_english_word_is_stripped_from_vietnamese_output():
    """Leading English discourse markers should be stripped if the rest is Vietnamese."""
    text = "unfortunately, tình hình trở nên khó khăn"
    assert LlmTranslator._clean_translation(text) == "tình hình trở nên khó khăn"


def test_leading_english_phrase_is_stripped_from_vietnamese_output():
    """Multiple leading English words should be stripped when Vietnamese follows."""
    text = "picks up the new layer ones, bất kể cái gì đó"
    assert LlmTranslator._clean_translation(text) == "bất kể cái gì đó"


def test_lowercase_leading_english_word_is_stripped_from_vietnamese_output():
    """Lowercase English words should be stripped if the remainder is Vietnamese."""
    text = "schedules như vậy đang được sử dụng."
    assert LlmTranslator._clean_translation(text) == "như vậy đang được sử dụng."


def test_sensitive_content_refusal():
    """LLM claiming sensitive content should be rejected."""
    text = "Nội dung nhạy cảm và không phù hợp"
    assert LlmTranslator._clean_translation(text) == ""


def test_vi_prefix_stripped():
    """VI: prefix should be stripped."""
    assert LlmTranslator._clean_translation("VI: Xin chào mọi người") == "Xin chào mọi người"


def test_meta_explanation_detected():
    """LLM explaining instead of translating should be caught."""
    text = "Tôi hiểu rồi, bạn muốn tôi dịch đoạn hội thoại này từ tiếng Nhật sang tiếng Việt."
    assert LlmTranslator._clean_translation(text) == ""


def test_conversational_response_with_preamble():
    """Model responding conversationally with 'bạn muốn tôi' should be caught."""
    text = "Bạn muốn tôi dịch câu này sang tiếng Việt phải không?"
    assert LlmTranslator._clean_translation(text) == ""


def test_meta_ai_assistant_response_returns_empty():
    """AI-assistant style responses should be rejected."""
    text = "Tôi là trợ lý AI. Tôi có thể giúp bạn."
    assert LlmTranslator._clean_translation(text) == ""


def test_multiline_meta_response_skips_to_translation():
    """Multi-line meta responses should be skipped like preambles."""
    text = "Tôi là trợ lý AI.\nXin chào mọi người"
    assert LlmTranslator._clean_translation(text) == "Xin chào mọi người"
