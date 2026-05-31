"""Regression tests for sentence-aware translation (no dropped sentences).

NLLB-600M silently drops trailing sentences when handed a multi-sentence block;
the translator now segments first and translates each sentence. These tests lock
in both the pure splitter (no model required) and the end-to-end behaviour (model
required) so the data-loss bug cannot silently return.

Run from the project root:

    python3 tests/test_translator_sentence_split.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.sentence_aggregator import split_japanese_sentences  # noqa: E402


class SentenceSplitTest(unittest.TestCase):
    def test_punctuated_multi_sentence_splits(self) -> None:
        self.assertEqual(
            split_japanese_sentences("本日の会議を始めます。資料を確認してください。"),
            ["本日の会議を始めます。", "資料を確認してください。"],
        )

    def test_terminal_run_on_splits(self) -> None:
        self.assertEqual(
            split_japanese_sentences("じゃあ時間を変更しておいてくださいはい分かりました"),
            ["じゃあ時間を変更しておいてください", "はい分かりました"],
        )

    def test_greeting_run_on_splits(self) -> None:
        # Reproduced data-loss case: a greeting run-on with no internal
        # punctuation reached NLLB as one block and the trailing turns were
        # dropped. Greeting boundaries (こんにちは / ようこそ) must split it.
        self.assertEqual(
            split_japanese_sentences(
                "皆さんこんにちは日本語ポッドキャストへようこそ皆さんお元気ですか"
            ),
            ["皆さんこんにちは", "日本語ポッドキャストへようこそ", "皆さんお元気ですか"],
        )

    def test_greeting_with_continuation_not_split(self) -> None:
        # こんにちは*と*言いました is one sentence (quoted greeting): no split.
        self.assertEqual(
            split_japanese_sentences("こんにちはと言いました"),
            ["こんにちはと言いました"],
        )

    def test_final_particle_does_not_split_mid_word(self) -> None:
        # FIX 3 regression: よ/わ/さ/ね must only be consumed as sentence-final
        # particles when a sentence genuinely ends there. They previously split
        # mid-word (e.g. ますよ|ろしく), corrupting both halves.
        self.assertEqual(
            split_japanese_sentences("始めますよろしくお願いします"),
            ["始めます", "よろしくお願いします"],
        )
        self.assertEqual(
            split_japanese_sentences("大丈夫ですわかりました"),
            ["大丈夫です", "わかりました"],
        )

    def test_final_particle_kept_when_truly_final(self) -> None:
        # …ますよ at end of buffer IS a final particle: keep it, do not split.
        self.assertEqual(split_japanese_sentences("行きますよ"), ["行きますよ"])

    def test_terminal_followed_by_connective_splits(self) -> None:
        # ですね/ですか followed by a sentence-starter must split AFTER the
        # particle, not swallow the connective or split mid-word.
        self.assertEqual(
            split_japanese_sentences("今日は晴れですねでは始めましょう"),
            ["今日は晴れですね", "では始めましょう"],
        )
        self.assertEqual(
            split_japanese_sentences("北海道ですかそれはいいですね"),
            ["北海道ですか", "それはいいですね"],
        )

    def test_greeting_followed_by_starter_splits(self) -> None:
        # B6: では/でも/はい/もう may start a sentence right after a greeting;
        # the continuation guard must NOT treat their leading kana as a particle.
        self.assertEqual(
            split_japanese_sentences("こんにちはでは始めます"),
            ["こんにちは", "では始めます"],
        )

    def test_single_sentence_returns_itself(self) -> None:
        self.assertEqual(split_japanese_sentences("本日の会議を始めます。"), ["本日の会議を始めます。"])
        self.assertEqual(split_japanese_sentences("こんにちは"), ["こんにちは"])

    def test_empty_returns_empty_list(self) -> None:
        self.assertEqual(split_japanese_sentences(""), [])
        self.assertEqual(split_japanese_sentences("   "), [])


@unittest.skipUnless(Path(config.NLLB_CT2_DIR).exists(), f"missing model dir: {config.NLLB_CT2_DIR}")
class SentenceAwareTranslationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.translator import NllbTranslator

        cls.translator = NllbTranslator()

    def test_no_trailing_sentence_dropped(self) -> None:
        # Both sentences must be represented; the bug dropped the second one.
        vietnamese = self.translator.translate("本日の会議を始めます。資料を確認してください。")
        print(f"two-sentence: {vietnamese}")
        lowered = vietnamese.casefold()
        self.assertIn("họp", lowered)            # from sentence 1
        self.assertIn("tài liệu", lowered)       # from sentence 2 (previously dropped)

    def test_translate_many_aligns_outputs(self) -> None:
        out = self.translator.translate_many(["はい。", "", "ありがとうございます。"])
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0])
        self.assertEqual(out[1], "")  # empty input maps to empty output, position preserved
        self.assertTrue(out[2])

    def test_single_sentence_unchanged(self) -> None:
        self.assertTrue(self.translator.translate("本日の会議を始めます。").strip())


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
