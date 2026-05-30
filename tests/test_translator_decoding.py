"""Regression tests for NLLB decoding quality knobs.

Run from the project root:

    python3 tests/test_translator_decoding.py
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.translator import NllbTranslator  # noqa: E402

WORD_RE = re.compile(r"\w+", re.UNICODE)


def has_immediately_repeated_bigram(text: str) -> bool:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    return any(words[i : i + 2] == words[i + 2 : i + 4] for i in range(len(words) - 3))


@unittest.skipUnless(Path(config.NLLB_CT2_DIR).exists(), f"missing model dir: {config.NLLB_CT2_DIR}")
class TranslatorDecodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.translator = NllbTranslator()

    def test_japanese_translates_to_non_empty_vietnamese(self) -> None:
        vietnamese = self.translator.translate("本日の会議を始めます。資料を確認してください。").strip()
        print(f"non-empty sample: {vietnamese}")
        self.assertTrue(vietnamese)

    def test_repetition_prone_input_has_no_degenerate_bigram_loop(self) -> None:
        vietnamese = self.translator.translate("申し訳ありません。少しお待ちください。もう一度説明します。").strip()
        print(f"repetition sample: {vietnamese}")
        self.assertTrue(vietnamese)
        self.assertFalse(has_immediately_repeated_bigram(vietnamese), vietnamese)


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
