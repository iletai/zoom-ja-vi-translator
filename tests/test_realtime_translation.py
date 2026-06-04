"""Live translation test: real Japanese domain sentences through NLLB + LLM.

Tests actual translation output quality on IT/AWS, rescue/emergency, and
hospital domain text. Requires models to be downloaded.

Usage: python tests/test_realtime_translation.py [--llm]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Real domain sentences for testing
TEST_SENTENCES = {
    "IT / AWS": [
        "本番環境のAWSのLambdaがタイムアウトしています",
        "EC2インスタンスのスケーリングポリシーを見直す必要があります",
        "S3バケットのパーミッションをクロスアカウントで設定してください",
        "APIゲートウェイのレスポンスが遅いのでCloudFrontを検討します",
        "マルチテナントのデータベース設計でテナント分離が重要です",
        "クロステナントのアクセス制御をロールベースで実装する",
        "ステージング環境にデプロイしてからプロダクションに反映します",
        "マイクロサービス間の通信はAPIで行います",
    ],
    "救急搬送・消防 (Rescue/Fire)": [
        "救急車が現場に到着し傷病者を搬送します",
        "消防署から救急隊が出動しました",
        "搬送先の病院が受入可能か交渉状態を確認してください",
        "多数傷病者が発生した場合は広域地図で搬送先を振り分けます",
        "病院連携システムで交渉履歴を確認できます",
        "搬送決定後に引き継ぎ資料を作成します",
        "救急搬送の出動から受入までの流れを説明します",
    ],
    "Mixed IT + Hospital System": [
        "消防側のテナントから病院テナントへクロステナントでデータを送信する",
        "救急搬送システムのバックエンドをAWSに移行する計画です",
        "交渉開始のAPIを呼び出すとき消防テナントの認証が必要です",
        "ユースケースとして搬送者情報の照会と受入確認があります",
        "病院側でも交渉状態とステータスをリアルタイムで更新する",
        "クロステナントの病院がデータを取得するときのパーミッション設計",
    ],
}

# Expected keywords that MUST appear in translation output for key terms
KEYWORD_CHECKS = {
    "テナント": ["tenant", "Tenant"],
    "クロステナント": ["Cross-Tenant"],
    "マルチテナント": ["Multi-Tenant"],
    "消防": ["cứu hỏa"],
    "引き継ぎ": ["bàn giao"],
    "病院連携": ["bệnh viện"],
    "ユースケース": ["Use-Case"],
}


def test_nllb():
    """Test NLLB translator with domain sentences."""
    print("\n" + "=" * 80)
    print("  NLLB-600M Translation Test (with glossary)")
    print("=" * 80)

    try:
        from src.translator import NllbTranslator
        translator = NllbTranslator()
    except Exception as e:
        print(f"  ERROR: Cannot load NLLB: {e}")
        return False

    total_time = 0
    results = []
    keyword_hits = 0
    keyword_total = 0

    for category, sentences in TEST_SENTENCES.items():
        print(f"\n─── {category} ───")
        for jp in sentences:
            start = time.time()
            vi = translator.translate(jp)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            results.append((jp, vi, elapsed))

            print(f"  [{elapsed:5.0f}ms] {jp}")
            print(f"         → {vi}")

            # Check keywords
            for jp_key, expected_kws in KEYWORD_CHECKS.items():
                if jp_key in jp:
                    keyword_total += 1
                    vi_lower = vi.lower()
                    if all(kw.lower() in vi_lower for kw in expected_kws):
                        keyword_hits += 1
                    else:
                        missing = [k for k in expected_kws if k.lower() not in vi_lower]
                        print(f"         ⚠ Missing: {missing}")

    # Summary
    print(f"\n{'─' * 80}")
    n = len(results)
    avg_ms = total_time / n if n else 0
    latencies = sorted(r[2] for r in results)
    p50 = latencies[n // 2] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    print(f"  Sentences: {n} | Avg: {avg_ms:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")
    print(f"  Keyword accuracy: {keyword_hits}/{keyword_total} ({100*keyword_hits/keyword_total:.0f}%)" if keyword_total else "")
    return keyword_hits, keyword_total


def test_llm():
    """Test LLM translator with domain sentences."""
    print("\n" + "=" * 80)
    print("  Qwen2.5-3B LLM Translation Test")
    print("=" * 80)

    try:
        from src.llm_translator import LlmTranslator
        translator = LlmTranslator()
    except Exception as e:
        print(f"  ERROR: Cannot load LLM: {e}")
        return False

    total_time = 0
    results = []
    keyword_hits = 0
    keyword_total = 0

    for category, sentences in TEST_SENTENCES.items():
        print(f"\n─── {category} ───")
        for jp in sentences:
            start = time.time()
            vi = translator.translate(jp)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            results.append((jp, vi, elapsed))

            print(f"  [{elapsed:5.0f}ms] {jp}")
            print(f"         → {vi}")

            # Check keywords
            for jp_key, expected_kws in KEYWORD_CHECKS.items():
                if jp_key in jp:
                    keyword_total += 1
                    vi_lower = vi.lower()
                    if all(kw.lower() in vi_lower for kw in expected_kws):
                        keyword_hits += 1
                    else:
                        missing = [k for k in expected_kws if k.lower() not in vi_lower]
                        print(f"         ⚠ Missing: {missing}")

    # Summary
    print(f"\n{'─' * 80}")
    n = len(results)
    avg_ms = total_time / n if n else 0
    latencies = sorted(r[2] for r in results)
    p50 = latencies[n // 2] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    print(f"  Sentences: {n} | Avg: {avg_ms:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")
    print(f"  Keyword accuracy: {keyword_hits}/{keyword_total} ({100*keyword_hits/keyword_total:.0f}%)" if keyword_total else "")
    return keyword_hits, keyword_total


if __name__ == "__main__":
    use_llm = "--llm" in sys.argv

    nllb_result = test_nllb()

    if use_llm:
        llm_result = test_llm()
    else:
        print("\n  (Run with --llm to also test Qwen2.5 LLM backend)")
        llm_result = None

    # Final verdict
    print("\n" + "=" * 80)
    if nllb_result:
        hits, total = nllb_result
        pct = 100 * hits / total if total else 0
        if pct >= 80:
            print(f"RESULT: PASS (NLLB keyword accuracy: {pct:.0f}%)")
        else:
            print(f"RESULT: FAIL (NLLB keyword accuracy: {pct:.0f}% < 80%)")
            sys.exit(1)
    else:
        print("RESULT: FAIL (could not load translator)")
        sys.exit(1)
