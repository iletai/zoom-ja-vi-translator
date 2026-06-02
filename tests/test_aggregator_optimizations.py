"""Tests for sentence aggregator optimization features.

Tests cover:
- New _CONTINUATIONS entries (みたいな, ような, ということ, etc.)
- ends_with_connective() method
- Pipeline-level connective buffering behavior
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentence_aggregator import SentenceAggregator, split_japanese_sentences  # noqa: E402


# ──── New Continuation Tests ──────────────────────────────────────────────────


class TestNewContinuations:
    """Tests for newly added continuation patterns that prevent splitting."""

    def test_mitai_na_prevents_split(self):
        """しましょうみたいな should NOT split (the original bug)."""
        agg = SentenceAggregator()
        text = "作って設計をしましょうみたいな話が上がってた"
        result = agg.add(text)
        assert result == [], f"Should not split at しましょうみたいな, got: {result}"
        assert agg.pending() == text

    def test_mitai_ni_prevents_split(self):
        """ですみたいに should NOT split."""
        agg = SentenceAggregator()
        text = "完了ですみたいに報告してください"
        result = agg.add(text)
        assert result == []
        assert agg.pending() == text

    def test_mitai_de_prevents_split(self):
        """ましたみたいで should NOT split."""
        agg = SentenceAggregator()
        text = "終わりましたみたいで安心しました"
        result = agg.add(text)
        # Should not split at ましたみたいで
        assert "終わりましたみたいで安心しました" in (result + [agg.pending()])

    def test_mitai_alone_prevents_split(self):
        """ですみたい should NOT split."""
        agg = SentenceAggregator()
        text = "大丈夫ですみたい"
        result = agg.add(text)
        assert result == []

    def test_you_na_prevents_split(self):
        """ましたような should NOT split."""
        agg = SentenceAggregator()
        text = "確認しましたような気がします"
        result = agg.add(text)
        assert result == []

    def test_to_iu_koto_prevents_split(self):
        """ですということ should NOT split."""
        agg = SentenceAggregator()
        text = "問題ですということは分かっています"
        result = agg.add(text)
        assert result == []

    def test_tte_iu_prevents_split(self):
        """ましたっていう should NOT split."""
        agg = SentenceAggregator()
        text = "完了しましたっていう報告がありました"
        result = agg.add(text)
        assert result == []

    def test_tte_kanji_prevents_split(self):
        """ましたって感じ should NOT split."""
        agg = SentenceAggregator()
        text = "終わりましたって感じです"
        result = agg.add(text)
        assert result == []

    def test_normal_split_still_works(self):
        """Regular splits should still function (no regression)."""
        agg = SentenceAggregator()
        # しましょう followed by a new clause (not a continuation)
        text = "設計をしましょう次のタスクは"
        result = agg.add(text)
        assert "設計をしましょう" in result

    def test_split_japanese_sentences_preserves_continuations(self):
        """split_japanese_sentences should keep continuations together."""
        text = "作って設計をしましょうみたいな話が上がってたかなと思うんですけど"
        sentences = split_japanese_sentences(text)
        # Should NOT be split at しましょう
        assert not any(s == "作って設計をしましょう" for s in sentences)
        # The full text should be present in the output
        combined = "".join(sentences)
        assert combined == text

    def test_original_continuations_still_work(self):
        """Original continuations (けど, って, ので, etc.) should still prevent split."""
        agg = SentenceAggregator()
        for text in (
            "大丈夫ですけど心配です",
            "言いましたって聞きました",
            "終わりましたので報告します",
            "確認しますからお待ちください",
        ):
            agg.reset()
            result = agg.add(text)
            assert result == [], f"Should not split: {text}, got: {result}"


# ──── ends_with_connective Tests ─────────────────────────────────────────────


class TestEndsWithConnective:
    """Tests for the ends_with_connective method."""

    def test_dattari_de(self):
        assert SentenceAggregator().ends_with_connective("スキースだったりで")

    def test_tari_de(self):
        assert SentenceAggregator().ends_with_connective("確認したりで")

    def test_tame_ni(self):
        assert SentenceAggregator().ends_with_connective("理解を合わせるために")

    def test_node(self):
        assert SentenceAggregator().ends_with_connective("検知していただいたので")

    def test_kara(self):
        assert SentenceAggregator().ends_with_connective("始めてから")

    def test_kedo(self):
        assert SentenceAggregator().ends_with_connective("思うんですけど")

    def test_keredomo(self):
        assert SentenceAggregator().ends_with_connective("問題ですけれども")

    def test_nagara(self):
        assert SentenceAggregator().ends_with_connective("走りながら")

    def test_tsutsu(self):
        assert SentenceAggregator().ends_with_connective("確認しつつ")

    def test_tari(self):
        assert SentenceAggregator().ends_with_connective("食べたり")

    def test_toshite(self):
        assert SentenceAggregator().ends_with_connective("前提として")

    def test_ni_taishite(self):
        assert SentenceAggregator().ends_with_connective("この問題に対して")

    def test_ni_tsuite(self):
        assert SentenceAggregator().ends_with_connective("この件について")

    def test_ni_yotte(self):
        assert SentenceAggregator().ends_with_connective("変更によって")

    def test_ni_kanshite(self):
        assert SentenceAggregator().ends_with_connective("品質に関して")

    def test_desu_ga(self):
        assert SentenceAggregator().ends_with_connective("問題なんですが")

    def test_desu_kedo(self):
        assert SentenceAggregator().ends_with_connective("難しいんですけど")

    def test_complete_sentence_not_connective(self):
        """Complete sentences should NOT be flagged as connective."""
        agg = SentenceAggregator()
        assert not agg.ends_with_connective("確認しました")
        assert not agg.ends_with_connective("設計をしましょう")
        assert not agg.ends_with_connective("APIを修正します")
        assert not agg.ends_with_connective("分かりました")

    def test_short_text_not_detected(self):
        """Very short text (< 4 chars) should not be detected."""
        agg = SentenceAggregator()
        assert not agg.ends_with_connective("で")
        assert not agg.ends_with_connective("て")
        assert not agg.ends_with_connective("")

    def test_filler_not_connective(self):
        """Fillers should not be detected as connectives."""
        agg = SentenceAggregator()
        assert not agg.ends_with_connective("はい")
        assert not agg.ends_with_connective("うん")
        assert not agg.ends_with_connective("なるほど")


# ──── Dangling Fragment Tests ─────────────────────────────────────────────────


class TestDanglingExtended:
    """Tests verifying is_dangling works with new entries."""

    def test_mitai_na_is_dangling(self):
        agg = SentenceAggregator()
        assert agg.is_dangling("みたいな")

    def test_ppoi_is_dangling(self):
        agg = SentenceAggregator()
        assert agg.is_dangling("っぽい")

    def test_to_iu_koto_is_dangling(self):
        agg = SentenceAggregator()
        assert agg.is_dangling("ということ")

    def test_to_iu_koto_de_is_dangling(self):
        agg = SentenceAggregator()
        assert agg.is_dangling("ということで")

    def test_nan_desu_kedo_is_dangling(self):
        agg = SentenceAggregator()
        assert agg.is_dangling("なんですけど")

    def test_longer_text_not_dangling(self):
        """Longer meaningful text should NOT be dangling."""
        agg = SentenceAggregator()
        assert not agg.is_dangling("確認しました")
        assert not agg.is_dangling("スキースだったりで")


# ──── Split Regression Tests ─────────────────────────────────────────────────


class TestSplitRegressions:
    """Regression tests from actual log data."""

    def test_seq39_no_split_at_shimashou_mitai_na(self):
        """Reproduce seq #39 bug: しましょうみたいな should not split."""
        text = "作って設計をしましょうみたいな話が上がってたかなと思うんですけど"
        sentences = split_japanese_sentences(text)
        # Must NOT have isolated "作って設計をしましょう" as a sentence
        for s in sentences:
            assert s != "作って設計をしましょう", \
                f"Should not split at しましょうみたいな: {sentences}"

    def test_normal_mashou_still_splits(self):
        """しましょう at a real boundary should still split."""
        agg = SentenceAggregator()
        text = "始めましょうでは次の話題です"
        result = agg.add(text)
        # "始めましょう" should be a separate sentence since "では" is a starter
        assert any("始めましょう" in s for s in result)

    def test_no_text_loss_in_continuation_split(self):
        """No text should be lost when continuations prevent splitting."""
        text = "確認しましたみたいな感じで進めましょう。"
        sentences = split_japanese_sentences(text)
        combined = "".join(sentences)
        assert combined == text, f"Text loss: {combined!r} != {text!r}"
