"""Fast unit tests for the source-side domain glossary (no model load)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.translator import NllbTranslator  # noqa: E402


def test_glossary_replaces_known_terms() -> None:
    out = NllbTranslator._apply_glossary("新幹線で北海道へ箱根経由")
    assert "新幹線" not in out and "北海道" not in out and "箱根" not in out
    assert "tàu Shinkansen" in out
    assert "tỉnh Hokkaido" in out
    assert "Hakone" in out


def test_glossary_leaves_other_text_untouched() -> None:
    text = "本日の会議を始めます"
    assert NllbTranslator._apply_glossary(text) == text


def test_glossary_entries_are_configured() -> None:
    assert config.NLLB_GLOSSARY.get("新幹線")
    assert config.NLLB_GLOSSARY.get("箱根")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n=== RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(tests) - failed}/{len(tests)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
