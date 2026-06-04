"""Test post-ASR correction module.

Verifies that domain-specific text corrections are applied correctly
to common ASR misrecognition patterns from real meeting logs.

Usage: python tests/test_post_correction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.post_correction import post_correct


def test_phrase_corrections():
    """Test exact phrase replacements."""
    cases = [
        # (input, expected_output)
        ("口臭状態が更新されました", "交渉状態が更新されました"),
        ("口臭履歴を確認", "交渉履歴を確認"),
        ("パミッションの設定", "パーミッションの設定"),
        ("マスト環境でテスト", "テスト環境でテスト"),
        ("デブ二にデプロイ", "dev2にデプロイ"),
        ("クロス祖母のアクセス", "クロステナントのアクセス"),
        ("マルチ祖母アーキテクチャ", "マルチテナントアーキテクチャ"),
        ("ペナントの権限", "テナントの権限"),
    ]
    passed = 0
    for inp, expected in cases:
        result = post_correct(inp)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: post_correct({inp!r})")
            print(f"    expected: {expected!r}")
            print(f"    got:      {result!r}")
    print(f"Phrase corrections: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_context_corrections():
    """Test context-aware regex corrections."""
    cases = [
        # 祖母 → 消防 when followed by system-related suffixes
        ("祖母の出動指令", "消防の出動指令"),
        ("祖母署に連絡", "消防署に連絡"),
        ("祖母隊が到着", "消防隊が到着"),
        # 昭和 → 消防 when NOT followed by era markers
        ("昭和テナントのデータ", "消防テナントのデータ"),
        # 昭和 should NOT be corrected when it's an era reference
        ("昭和時代の建物", "昭和時代の建物"),
        ("昭和30年", "昭和30年"),
        # スケース → ユースケース when not preceded by ユー
        ("スケースを確認", "ユースケースを確認"),
    ]
    passed = 0
    for inp, expected in cases:
        result = post_correct(inp)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: post_correct({inp!r})")
            print(f"    expected: {expected!r}")
            print(f"    got:      {result!r}")
    print(f"Context corrections: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_no_false_positives():
    """Test that correct text is not modified."""
    cases = [
        "テナントの設定を変更します",
        "消防署からの連絡",
        "交渉状態を確認",
        "パーミッションが必要",
        "テスト環境にデプロイ",
        "ユースケースを定義",
        "",
        "普通のテキスト",
    ]
    passed = 0
    for inp in cases:
        result = post_correct(inp)
        if result == inp:
            passed += 1
        else:
            print(f"  FAIL: post_correct({inp!r}) modified to {result!r}")
    print(f"No false positives: {passed}/{len(cases)} passed")
    return passed == len(cases)


if __name__ == "__main__":
    print("=== Test Post-ASR Correction ===\n")
    results = [
        test_phrase_corrections(),
        test_context_corrections(),
        test_no_false_positives(),
    ]
    print()
    if all(results):
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)
