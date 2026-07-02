"""Integration test: domain accuracy using real meeting log inputs.

Tests the full post-correction + glossary pipeline against actual Japanese text
captured from the 2026-06-04 meeting (救急搬送システム IT project discussion).

Input texts are REAL ASR outputs from test_audio/evidence/run_20260604_113106.jsonl.
This test validates:
  1. Post-ASR correction fixes known misrecognitions
  2. NLLB glossary substitution produces correct domain terms
  3. No regressions on clean text (false positives)

Usage: python tests/test_meeting_accuracy.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.post_correction import post_correct

# NllbTranslator imported lazily in test_translation_keywords() since it
# requires ctranslate2 which may not be installed in CI/test environments.


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Post-ASR Correction Tests (no model needed)
# ═══════════════════════════════════════════════════════════════════════════════

# Real ASR outputs from meeting log that contain known misrecognitions.
# Format: (raw_asr_output, expected_after_correction)
POST_CORRECTION_CASES = [
    # Event #14: デブ二 → dev2
    (
        "デブ二からテスト環境にマージ分はできるかぎりチェックできるように",
        "dev2からテスト環境にマージ分はできるかぎりチェックできるように",
    ),
    # Event #16: マスト環境 → テスト環境
    (
        "マスト環境のほうで確認できる状態となっています",
        "テスト環境のほうで確認できる状態となっています",
    ),
    # Event #102: 口臭状態 → 交渉状態
    (
        "病院側でも口臭状態と",
        "病院側でも交渉状態と",
    ),
    # Event #131: 祖母側 → 消防側 (context: followed by の)
    (
        "祖母側のほうは恐らく病院センターの画面のほうが主なところになっていてその辺の注意をして",
        "消防側のほうは恐らく病院センターの画面のほうが主なところになっていてその辺の注意をして",
    ),
    # Event #142: 昭和 → 消防, ペナント → テナント
    (
        "昭和がほかのペナントの見えるかもしれないと思います",
        "消防がほかのテナントの見えるかもしれないと思います",
    ),
    # Event #122: パミッション → パーミッション
    (
        "パミッションで心とか五定義できればいいのかなというので",
        "パーミッションで心とか五定義できればいいのかなというので",
    ),
    # Compound: クロス祖母 → クロステナント
    (
        "クロス祖母のデータを取得する",
        "クロステナントのデータを取得する",
    ),
    # Compound: マルチ祖母 → マルチテナント
    (
        "マルチ祖母のアーキテクチャ",
        "マルチテナントのアーキテクチャ",
    ),
    # Event #37: 祖母側 (context: no suffix match → should NOT correct)
    # Note: "祖母側" doesn't match the regex (?=の|署|隊|車|局), so won't be corrected
    # by context pattern. But "祖母側" contains 祖母 followed by 側 which is not in our
    # context list. This tests that we DON'T over-correct.
]

# Real ASR outputs that are CORRECT and should NOT be modified
CLEAN_TEXTS = [
    # Event #83: correctly recognized domain terms
    "消防側のテナントで処理が",
    # Event #84: クロステナント correct
    "あとにクロステナントにこういう",
    # Event #98: 搬送決定 correct
    "搬送決定したよっていう時にほかの病院の搬送決定も消すよとか",
    # Event #99: ユースケース correct
    "そういうユースケースを網羅してその時にステータス更新は",
    # Event #103: 交渉状態 + クロステナント correct
    "消防が持っている交渉状態とクロステナントの病院でっていうちょっと受けも来れないよねっていうとこは",
    # Event #105: 消防 correct
    "消防がキーになってて",
    # Event #109: クロステナント correct
    "この時はクロステナントに取りに行くことはないです",
    # Event #115: 消防 correct
    "はい消防視点で",
    # Event #129: 病院連携 correct
    "病院連携業務のほうでやり取りが発生する時に騒然する可能性はあるかもしれないとかですね",
    # Event #149: クロステナント correct
    "そこはクロステナントの要件には入ってないんで出さないで地図だけのところあっ",
    # Event #152: 傷病者 correct
    "いろんな病院に搬送するパターンですけどどっち傷病者の",
    # Event #161: 搬送元 + 消防車 correct
    "搬送元の消防車の情報見ればだめとというところです",
    # Event #399: short correct
    "消防のビデオ",
    # General text without domain terms
    "資料はチャットの方でお送りいたします",
    "動作確認を行っている最中です",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Glossary Substitution Tests (no model needed, tests config logic)
# ═══════════════════════════════════════════════════════════════════════════════

# Test that glossary entries in config.NLLB_GLOSSARY properly match expected terms.
# These are the key domain terms that MUST be in the glossary.
REQUIRED_GLOSSARY_ENTRIES = {
    "テナント": "Tenant",
    "クロステナント": "Cross-Tenant",
    "マルチテナント": "Multi-Tenant",
    "ユースケース": "Use-Case",
    "デプロイ": "Deploy",
    "マイクロサービス": "Microservice",
    "ステージング": "Staging",
}

# Test that glossary applies longest-match-first (クロステナント before テナント)
GLOSSARY_ORDERING_CASES = [
    # Input text → term that SHOULD be substituted (longest match wins)
    ("クロステナントのデータ", "Cross-Tenant"),
    ("マルチテナントの設計", "Multi-Tenant"),
    ("テナントの権限", "Tenant"),
    ("ユースケースを定義", "Use-Case"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Translation Quality Tests (requires NLLB model)
# ═══════════════════════════════════════════════════════════════════════════════

# Real meeting inputs after post-correction → expected Vietnamese keywords
# (not exact match — just check that key domain terms appear in output)
TRANSLATION_KEYWORD_CASES = [
    # Input (after correction) → list of Vietnamese keywords expected in output
    (
        "消防側のテナントで処理が",
        ["cứu hỏa"],  # fire dept term should appear
    ),
    (
        "クロステナントの要件には入ってないんで",
        ["Cross-Tenant"],
    ),
    (
        "搬送決定したよっていう時にほかの病院の搬送決定も消すよとか",
        ["bệnh viện"],  # hospital should appear
    ),
    (
        "ユースケースを網羅して",
        ["Use-Case"],
    ),
    (
        "消防がキーになってて",
        ["cứu hỏa"],
    ),
    (
        "引き継ぎの資料を準備",
        ["chuyển giao"],  # NLLB renders 引き継ぎ as "chuyển giao" via handover glossary
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def test_post_correction():
    """Test post-ASR corrections on real meeting text."""
    for inp, expected in POST_CORRECTION_CASES:
        result = post_correct(inp)
        assert result == expected, f"post_correct({inp!r}) → {result!r}, expected {expected!r}"


def test_no_false_positives():
    """Test that correctly recognized text is NOT modified."""
    for text in CLEAN_TEXTS:
        result = post_correct(text)
        assert result == text, f"post_correct({text!r}) was modified to {result!r}"


def test_glossary_entries():
    """Test that all required domain terms exist in NLLB_GLOSSARY."""
    for jp_term, expected_replacement in REQUIRED_GLOSSARY_ENTRIES.items():
        actual = config.NLLB_GLOSSARY.get(jp_term)
        assert actual == expected_replacement, (
            f"NLLB_GLOSSARY[{jp_term!r}] = {actual!r}, expected {expected_replacement!r}"
        )


def test_glossary_ordering():
    """Test that glossary applies longest-match-first."""
    sorted_keys = sorted(config.NLLB_GLOSSARY.keys(), key=len, reverse=True)
    for text, expected_sub in GLOSSARY_ORDERING_CASES:
        result = text
        for jp_key in sorted_keys:
            if jp_key in result:
                result = result.replace(jp_key, config.NLLB_GLOSSARY[jp_key])
        assert expected_sub in result, f"{text!r} → {result!r}, missing {expected_sub!r}"


def test_translation_keywords():
    """Test translation output contains expected domain keywords.

    Requires NLLB model to be downloaded. Skipped if model unavailable.
    """
    try:
        from src.translator import NllbTranslator
        translator = NllbTranslator()
    except Exception as e:
        pytest.skip(f"model not available: {e}")

    for jp_input, expected_keywords in TRANSLATION_KEYWORD_CASES:
        vi_output = translator.translate(jp_input)
        vi_lower = vi_output.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in vi_lower]
        assert not missing, (
            f"translate({jp_input!r}) → {vi_output!r}, missing keywords: {missing}"
        )


def test_config_parameters():
    """Test that performance parameters are set correctly."""
    checks = [
        ("NLLB_MAX_DECODING_LENGTH", config.NLLB_MAX_DECODING_LENGTH, 128),
        ("NLLB_NO_REPEAT_NGRAM_SIZE", config.NLLB_NO_REPEAT_NGRAM_SIZE, 4),
        ("NLLB_REPETITION_PENALTY", config.NLLB_REPETITION_PENALTY, 1.2),
        ("LLM_RAM_CACHE_MB", config.LLM_RAM_CACHE_MB, 256),
    ]
    for name, actual, expected in checks:
        assert actual == expected, f"{name} = {actual!r}, expected {expected!r}"


def test_post_translation_corrections():
    """Test that post-translation correction dict exists and has key entries."""
    corrections = getattr(config, "NLLB_POST_TRANSLATION", None)
    assert corrections is not None, "NLLB_POST_TRANSLATION not found in config"

    required = {
        "người thuê nhà": "tenant",
        "người thuê qua": "cross-tenant",
        "thừa kế thai nhi": "bàn giao",
        "thai nhi": "bàn giao",
    }
    for wrong, expected_right in required.items():
        actual = corrections.get(wrong)
        assert actual == expected_right, (
            f"NLLB_POST_TRANSLATION[{wrong!r}] = {actual!r}, expected {expected_right!r}"
        )


def test_hotwords_no_comments():
    """Test that hotwords_it.txt has no comment lines (sherpa-onnx bug)."""
    hotwords_path = Path("hotwords_it.txt")
    if not hotwords_path.is_file():
        pytest.skip("hotwords_it.txt not found")

    lines = hotwords_path.read_text(encoding="utf-8").splitlines()
    comment_lines = [i + 1 for i, l in enumerate(lines) if l.strip().startswith("#")]
    assert not comment_lines, (
        f"comment lines found at {comment_lines} — sherpa-onnx does NOT skip # lines"
    )
    bad_lines = [
        (i + 1, l.strip()) for i, l in enumerate(lines) if l.strip() and " :" not in l.strip()
    ]
    assert not bad_lines, f"malformed lines (missing ' :'): {bad_lines[:5]}"


if __name__ == "__main__":
    print("=" * 70)
    print("  Meeting Accuracy Test — 2026-06-04 救急搬送システム Meeting")
    print("=" * 70)
    print()

    results = []

    # Tests that don't need the model (always run)
    results.append(("Post-ASR Correction", test_post_correction()))
    print()
    results.append(("No False Positives", test_no_false_positives()))
    print()
    results.append(("Glossary Coverage", test_glossary_entries()))
    print()
    results.append(("Glossary Ordering", test_glossary_ordering()))
    print()
    results.append(("Config Parameters", test_config_parameters()))
    print()
    results.append(("Post-Translation Fixes", test_post_translation_corrections()))
    print()
    results.append(("Hotwords File", test_hotwords_no_comments()))
    print()

    # Model-dependent test (skipped if model not available)
    results.append(("Translation Keywords", test_translation_keywords()))
    print()

    # Summary
    print("=" * 70)
    all_pass = all(r[1] for r in results)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    print("=" * 70)
    print()

    if all_pass:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)
