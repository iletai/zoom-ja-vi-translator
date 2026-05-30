"""Unit tests for character-level LocalAgreement streaming commits."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_agreement import LocalAgreementBuffer  # noqa: E402


def test_local_agreement_2_commits_only_repeated_japanese_prefix() -> None:
    buf = LocalAgreementBuffer(n=2)
    sequence = ["こ", "こん", "こんに", "こんにち", "こんにちは"]
    expected = [
        ("", "こ"),
        ("こ", "ん"),
        ("こん", "に"),
        ("こんに", "ち"),
        ("こんにち", "は"),
    ]
    assert [buf.update(partial) for partial in sequence] == expected


def test_revision_before_commit_does_not_show_stale_characters() -> None:
    buf = LocalAgreementBuffer(n=2)
    partials = ["今日は", "明日は", "明日は会", "明日は会"]
    outputs = [buf.update(partial) for partial in partials]
    assert outputs == [
        ("", "今日は"),
        ("", "明日は"),
        ("明日は", "会"),
        ("明日は会", ""),
    ]
    committed_lengths = [len(committed) for committed, _ in outputs]
    assert committed_lengths == sorted(committed_lengths)
    for partial, (committed, tail) in zip(partials, outputs):
        assert partial == committed + tail


def test_committed_prefix_trims_if_asr_revises_after_commit() -> None:
    buf = LocalAgreementBuffer(n=2)
    assert buf.update("東京都") == ("", "東京都")
    assert buf.update("東京都") == ("東京都", "")
    assert buf.update("京都府") == ("", "京都府")


def test_n_greater_than_2_requires_prefix_stable_across_history() -> None:
    buf = LocalAgreementBuffer(n=3)
    assert buf.update("資料") == ("", "資料")
    assert buf.update("資料を") == ("", "資料を")
    assert buf.update("資料を") == ("資料", "を")
    assert buf.update("資料を共") == ("資料を", "共")


def test_n_1_commits_entire_hypothesis() -> None:
    buf = LocalAgreementBuffer(n=1)
    assert buf.update("会議") == ("会議", "")
    assert buf.update("会議を始めます") == ("会議を始めます", "")


def test_reset_clears_state() -> None:
    buf = LocalAgreementBuffer(n=2)
    assert buf.update("質問") == ("", "質問")
    assert buf.update("質問") == ("質問", "")
    buf.reset()
    assert buf.committed == ""
    assert buf.prev == ""
    assert buf.update("回答") == ("", "回答")


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
