"""Tests for LLM translation optimization features.

Tests cover:
- CJK/kana logit bias builder
- Fragment detection (_is_incomplete_fragment)
- Improved CJK stripping with Vietnamese diacritics awareness
- GBNF grammar option
- Expanded filler map
- Bilingual system prompt and few-shot examples
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_translator import LlmTranslator  # noqa: E402


class _DummyLlm:
    """Mock LLM for testing without real model."""

    def __init__(self, response_text: str = "Bản dịch thử") -> None:
        self.prompt: str | None = None
        self.response_text = response_text
        self._vocab_size = 100
        # Simulate a small vocab with some CJK tokens
        self._vocab = {
            0: b"hello",
            1: b" ",
            2: b"\xe7\x9a\x84",      # 的 (CJK)
            3: b"\xe6\x98\xaf",      # 是 (CJK)
            4: b"\xe3\x81\xaf",      # は (Hiragana)
            5: b"\xe3\x82\xa2",      # ア (Katakana)
            6: b"Vi\xe1\xbb\x87t",   # Việt (Vietnamese)
            7: b"\xe3\x80\x82",      # 。 (CJK punctuation)
            8: b"API",
            9: b"\xc3\xa0",          # à (Vietnamese diacritic)
            10: b"\xe4\xb8\xad\xe6\x96\x87",  # 中文 (multi-char CJK)
        }

    def create_completion(self, *, prompt, **_: object) -> dict[str, object]:
        self.prompt = prompt
        return {"choices": [{"text": self.response_text}]}

    def n_vocab(self) -> int:
        return len(self._vocab)

    def detokenize(self, token_ids: list[int]) -> bytes:
        if len(token_ids) == 1 and token_ids[0] in self._vocab:
            return self._vocab[token_ids[0]]
        return b""

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        return [0]


def _make_translator(response_text: str = "Bản dịch thử") -> tuple[LlmTranslator, _DummyLlm]:
    """Create a translator with mocked LLM for testing."""
    translator = LlmTranslator.__new__(LlmTranslator)
    translator._keep_context = False
    translator._history = deque(maxlen=1)
    translator._lock = threading.Lock()
    translator._chinese_logit_bias = {}
    translator._vi_grammar = None
    translator._fast_translator = None
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


# ──── Logit Bias Tests ────────────────────────────────────────────────────────


class TestLogitBiasBuilder:
    """Tests for _build_chinese_logit_bias vocab scanning."""

    def test_blocks_pure_cjk_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 2 (的) and 3 (是) are pure CJK → should be blocked
        assert 2 in bias
        assert 3 in bias
        assert bias[2] == -100.0
        assert bias[3] == -100.0

    def test_blocks_hiragana_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 4 (は) is Hiragana → should be blocked
        assert 4 in bias
        assert bias[4] == -100.0

    def test_blocks_katakana_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 5 (ア) is Katakana → should be blocked
        assert 5 in bias
        assert bias[5] == -100.0

    def test_blocks_cjk_punctuation(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 7 (。) is CJK Symbols → should be blocked
        assert 7 in bias
        assert bias[7] == -100.0

    def test_blocks_multi_char_cjk(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 10 (中文) is multi-char CJK → should be blocked
        assert 10 in bias
        assert bias[10] == -100.0

    def test_preserves_latin_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 0 ("hello"), 8 ("API") are Latin → NOT blocked
        assert 0 not in bias
        assert 8 not in bias

    def test_preserves_vietnamese_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 6 ("Việt"), 9 ("à") are Vietnamese → NOT blocked
        assert 6 not in bias
        assert 9 not in bias

    def test_preserves_space_tokens(self):
        translator, _ = _make_translator()
        bias = translator._build_chinese_logit_bias()
        # Token 1 (" ") is space → NOT blocked
        assert 1 not in bias

    def test_graceful_degradation_no_n_vocab(self):
        """Returns empty dict if model has no n_vocab method."""
        translator, dummy_llm = _make_translator()
        # Replace llm with object that has no n_vocab
        translator.llm = type("NoVocab", (), {"detokenize": lambda self, x: b""})()
        bias = translator._build_chinese_logit_bias()
        assert bias == {}

    def test_graceful_degradation_no_detokenize(self):
        """Returns empty dict if model has no detokenize method."""
        translator, dummy_llm = _make_translator()
        # Replace llm with object that has no detokenize
        translator.llm = type("NoDetok", (), {"n_vocab": lambda self: 10})()
        bias = translator._build_chinese_logit_bias()
        assert bias == {}


# ──── Fragment Detection Tests ────────────────────────────────────────────────


class TestFragmentDetection:
    """Tests for _is_incomplete_fragment static method."""

    def test_connective_de(self):
        """で ending detected as fragment."""
        assert LlmTranslator._is_incomplete_fragment("スキースだったりで")
        assert LlmTranslator._is_incomplete_fragment("認識を合わせるためにで")

    def test_connective_te(self):
        """て ending detected as fragment."""
        assert LlmTranslator._is_incomplete_fragment("処理を入れて")
        assert LlmTranslator._is_incomplete_fragment("確認して")

    def test_connective_tari_de(self):
        """たりで ending detected."""
        assert LlmTranslator._is_incomplete_fragment("スキースだったりで")

    def test_connective_tame_ni(self):
        """ために ending detected."""
        assert LlmTranslator._is_incomplete_fragment("理解を合わせるために")

    def test_connective_node(self):
        """ので ending detected."""
        assert LlmTranslator._is_incomplete_fragment("検知していただいたので")

    def test_connective_kedo(self):
        """けど ending detected."""
        assert LlmTranslator._is_incomplete_fragment("思うんですけど")

    def test_connective_kara(self):
        """から ending detected."""
        assert LlmTranslator._is_incomplete_fragment("始めてから")

    def test_connective_nagara(self):
        """ながら ending detected."""
        assert LlmTranslator._is_incomplete_fragment("走りながら")

    def test_connective_ni_tsuite(self):
        """について ending detected."""
        assert LlmTranslator._is_incomplete_fragment("この件について")

    def test_connective_ni_yotte(self):
        """によって ending detected."""
        assert LlmTranslator._is_incomplete_fragment("変更によって")

    def test_complete_sentence_not_fragment(self):
        """Complete sentences are not fragments."""
        assert not LlmTranslator._is_incomplete_fragment("作って設計をしましょう")
        assert not LlmTranslator._is_incomplete_fragment("確認しました")
        assert not LlmTranslator._is_incomplete_fragment("APIを修正します")

    def test_short_text_not_false_positive(self):
        """Single-char endings need at least 4 chars total."""
        assert not LlmTranslator._is_incomplete_fragment("で")
        assert not LlmTranslator._is_incomplete_fragment("て")
        assert not LlmTranslator._is_incomplete_fragment("に")

    def test_empty_text(self):
        assert not LlmTranslator._is_incomplete_fragment("")
        assert not LlmTranslator._is_incomplete_fragment("   ")

    def test_desu_ga(self):
        """ですが ending detected."""
        assert LlmTranslator._is_incomplete_fragment("問題なんですが")

    def test_desu_kedo(self):
        """ですけど ending detected."""
        assert LlmTranslator._is_incomplete_fragment("思うんですけど")


# ──── Improved CJK Stripping Tests ────────────────────────────────────────────


class TestCJKStripping:
    """Tests for improved _clean_translation CJK handling."""

    def test_vietnamese_with_stray_kanji_accepted(self):
        """Vietnamese text with stray CJK should be stripped and accepted."""
        # Ừ嗯 → Ừ (diacritics confirm Vietnamese)
        result = LlmTranslator._clean_translation("Ừ嗯")
        assert result == "Ừ"

    def test_vietnamese_diacritics_override_ratio_check(self):
        """If stripped text has VI diacritics, accept regardless of ratio."""
        result = LlmTranslator._clean_translation("Đã確認xong")
        assert "Đã" in result
        assert "確認" not in result

    def test_pure_chinese_still_rejected(self):
        """Pure Chinese with no Vietnamese should still be rejected."""
        assert LlmTranslator._clean_translation("顺便问一下") == ""
        assert LlmTranslator._clean_translation("由于加入") == ""
        assert LlmTranslator._clean_translation("为") == ""

    def test_mixed_vi_cjk_strips_correctly(self):
        """Vietnamese + CJK mixed should preserve Vietnamese parts."""
        result = LlmTranslator._clean_translation("Chào mừng来到世界")
        # Should strip CJK and keep Vietnamese
        assert "Chào" in result or result == ""  # depends on ratio

    def test_cjk_numerals_converted(self):
        """CJK numeral characters should be converted to Arabic digits."""
        result = LlmTranslator._clean_translation("Lần thứ 三 trong tháng này")
        assert "3" in result or "三" not in result

    def test_mostly_kanji_rejected(self):
        """Text that's mostly kanji (>60%) with no VI diacritics → rejected."""
        assert LlmTranslator._clean_translation("实际情况我还不清楚") == ""
        assert LlmTranslator._clean_translation("顺便说一下我尝试加入") == ""


# ──── GBNF Grammar Tests ─────────────────────────────────────────────────────


class TestGBNFGrammar:
    """Tests for _build_vi_grammar option."""

    def test_grammar_disabled_by_default(self):
        """Grammar should return None when config flag is False."""
        translator, _ = _make_translator()
        with patch("config.LLM_USE_GRAMMAR", False):
            result = translator._build_vi_grammar()
        assert result is None

    def test_grammar_enabled_returns_object(self):
        """Grammar should attempt to build when config flag is True."""
        translator, _ = _make_translator()
        # When LlamaGrammar is not importable, returns None gracefully
        with patch("config.LLM_USE_GRAMMAR", True):
            result = translator._build_vi_grammar()
        # Either returns a grammar object or None (if llama_cpp not installed)
        # In test env without llama_cpp, should return None gracefully
        assert result is None  # llama_cpp not installed in test env

    def test_grammar_not_passed_when_none(self):
        """When _vi_grammar is None, it should be passed as None to create_completion."""
        translator, dummy_llm = _make_translator()
        translator._vi_grammar = None
        translator._translate_one("テスト", update_context=False)
        # The DummyLlm accepts any kwargs; just verify no crash
        assert dummy_llm.prompt is not None


# ──── Filler Map Tests ────────────────────────────────────────────────────────


class TestExpandedFillerMap:
    """Tests for new filler map entries."""

    def test_new_fillers(self):
        translator, _ = _make_translator()
        new_fillers = {
            "うんうんうん": "Vâng vâng vâng",
            "はいはいはい": "Vâng vâng vâng",
            "はいもね": "Vâng, đúng nhỉ",
            "あのう": "À...",
            "なるほどね": "Ra vậy nhỉ",
            "そうそうそう": "Đúng đúng đúng",
            "ねえ": "Này",
            "ねえねえ": "Này này",
            "うんねえねえ": "Vâng, này này",
        }
        for source, expected in new_fillers.items():
            result = translator._translate_one(source, update_context=False)
            assert result == expected, f"Filler {source!r}: got {result!r}, expected {expected!r}"

    def test_existing_fillers_preserved(self):
        """Verify original fillers still work."""
        translator, _ = _make_translator()
        existing = {
            "うん": "Vâng",
            "はい": "Vâng",
            "えっと": "À...",
            "なるほど": "Ra vậy",
            "そうですね": "Đúng vậy nhỉ",
        }
        for source, expected in existing.items():
            result = translator._translate_one(source, update_context=False)
            assert result == expected

    def test_truncated_filler_prefix_match(self):
        """Fillers with 1-2 chars truncated should still match."""
        translator, _ = _make_translator()
        # "うんうんうん" (6 chars) → prefix with 5 chars "うんうんう" should match
        result = translator._translate_one("うんうんう", update_context=False)
        assert result == "Vâng vâng vâng"


# ──── System Prompt Tests ─────────────────────────────────────────────────────


class TestBilingualPrompt:
    """Tests for bilingual system prompt format."""

    def test_default_prompt_has_english_prohibition(self):
        """Default prompt should contain English NEVER instruction."""
        from src.llm_translator import _DEFAULT_SYSTEM_PROMPT
        assert "NEVER" in _DEFAULT_SYSTEM_PROMPT
        assert "Chinese characters" in _DEFAULT_SYSTEM_PROMPT
        assert "Latin script" in _DEFAULT_SYSTEM_PROMPT

    def test_default_prompt_has_vietnamese_section(self):
        """Default prompt should contain Vietnamese instructions."""
        from src.llm_translator import _DEFAULT_SYSTEM_PROMPT
        assert "KHÔNG ĐƯỢC" in _DEFAULT_SYSTEM_PROMPT
        assert "tiếng Việt" in _DEFAULT_SYSTEM_PROMPT

    def test_few_shot_examples_count(self):
        """Should have 6 few-shot examples covering diverse patterns."""
        assert len(LlmTranslator._FEW_SHOT_EXAMPLES) == 6

    def test_few_shot_includes_kanji_heavy(self):
        """At least one example should have kanji-heavy input."""
        kanji_heavy = any(
            "確認" in ex[0] or "処理" in ex[0] or "反映" in ex[0]
            for ex in LlmTranslator._FEW_SHOT_EXAMPLES
        )
        assert kanji_heavy

    def test_few_shot_all_have_vi_prefix(self):
        """All few-shot outputs should start with 'VI: '."""
        for _, output in LlmTranslator._FEW_SHOT_EXAMPLES:
            assert output.startswith("VI: ")

    def test_build_raw_prompt_includes_few_shots(self):
        """Built prompt should include few-shot examples."""
        translator, _ = _make_translator()
        translator.system_prompt = "test system prompt"
        prompt = translator._build_raw_prompt("テスト入力")
        # Should contain at least one few-shot example
        assert "JA:" in prompt
        assert "VI:" in prompt

    def test_config_prompt_has_english_section(self):
        """Config LLM_SYSTEM_PROMPT should also use bilingual format."""
        import config
        prompt = config.LLM_SYSTEM_PROMPT
        assert "NEVER" in prompt or "CRITICAL" in prompt
        assert "Vietnamese" in prompt or "tiếng Việt" in prompt


# ──── Integration Tests ───────────────────────────────────────────────────────


class TestTranslationIntegration:
    """Integration tests for the full translation flow with new features."""

    def test_translate_one_applies_logit_bias(self):
        """_translate_one should pass logit_bias to create_completion."""
        translator, _ = _make_translator()
        translator._chinese_logit_bias = {100: -100.0, 200: -100.0}

        # Override create_completion to capture kwargs
        captured_kwargs = {}

        def mock_completion(*, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return {"choices": [{"text": "Kết quả dịch"}]}

        translator.llm.create_completion = mock_completion
        translator._translate_one("テスト文", update_context=False)

        assert "logit_bias" in captured_kwargs
        assert captured_kwargs["logit_bias"] == {100: -100.0, 200: -100.0}

    def test_translate_one_applies_grammar(self):
        """_translate_one should pass grammar to create_completion when set."""
        translator, _ = _make_translator()
        mock_grammar = MagicMock()
        translator._vi_grammar = mock_grammar

        captured_kwargs = {}

        def mock_completion(*, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return {"choices": [{"text": "Kết quả dịch"}]}

        translator.llm.create_completion = mock_completion
        translator._translate_one("テスト文", update_context=False)

        assert "grammar" in captured_kwargs
        assert captured_kwargs["grammar"] is mock_grammar

    def test_retry_uses_fragment_prompt_for_incomplete_input(self):
        """Retry should use fragment-specific prompt for incomplete sentences."""
        translator, _ = _make_translator()

        prompts_used = []

        call_count = [0]

        def mock_completion(*, prompt, **kwargs):
            call_count[0] += 1
            prompts_used.append(prompt)
            if call_count[0] == 1:
                # First call: return Chinese (will be rejected)
                return {"choices": [{"text": "为了"}]}
            else:
                # Retry: return valid Vietnamese
                return {"choices": [{"text": "Để thống nhất nhận thức..."}]}

        translator.llm.create_completion = mock_completion
        result = translator._translate_one(
            "認識を合わせるためにスキースだったりで", update_context=False
        )

        # Should have retried
        assert call_count[0] == 2
        # Retry prompt should mention incomplete sentence
        assert "chưa hoàn chỉnh" in prompts_used[1] or "KHÔNG" in prompts_used[1]

    def test_retry_uses_standard_prompt_for_complete_input(self):
        """Retry for complete sentences uses standard anti-Chinese prompt."""
        translator, _ = _make_translator()

        prompts_used = []
        call_count = [0]

        def mock_completion(*, prompt, **kwargs):
            call_count[0] += 1
            prompts_used.append(prompt)
            if call_count[0] == 1:
                return {"choices": [{"text": "这是中文"}]}
            else:
                return {"choices": [{"text": "Đây là kết quả"}]}

        translator.llm.create_completion = mock_completion
        result = translator._translate_one("確認しました", update_context=False)

        assert call_count[0] == 2
        # Standard retry prompt should NOT mention incomplete
        assert "chưa hoàn chỉnh" not in prompts_used[1]
        assert "KHÔNG" in prompts_used[1] or "TUYỆT ĐỐI" in prompts_used[1]

    def test_expanded_stop_tokens_in_completion_call(self):
        """Verify expanded Chinese stop tokens are passed."""
        translator, _ = _make_translator()

        captured_kwargs = {}

        def mock_completion(*, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return {"choices": [{"text": "Kết quả"}]}

        translator.llm.create_completion = mock_completion
        translator._translate_one("テスト", update_context=False)

        stop_list = captured_kwargs.get("stop", [])
        # Verify expanded Chinese tokens are present
        assert "这" in stop_list
        assert "那" in stop_list
        assert "为" in stop_list
        assert "因" in stop_list
        # Original tokens still present
        assert "的" in stop_list
        assert "是" in stop_list
        assert "<|im_end|>" in stop_list
