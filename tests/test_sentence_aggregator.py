"""Script-style tests for streaming Japanese sentence aggregation."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentence_aggregator import SentenceAggregator, split_japanese_sentences  # noqa: E402


def test_mid_word_split_healed_by_merge() -> None:
    agg = SentenceAggregator()
    assert agg.add("では八時発に変") == []
    assert agg.pending() == "では八時発に変"
    assert agg.add("更しておきます") == []
    assert agg.flush() == ["では八時発に変更しておきます"]


def test_run_on_re_split_at_meeting_terminals() -> None:
    agg = SentenceAggregator()
    out = agg.add("じゃあ時間を変更しておいてくださいはい分かりましたでは八時発に")
    assert out == ["じゃあ時間を変更しておいてください", "はい分かりました"]
    assert agg.add("変更しておきます") == []
    assert agg.flush() == ["では八時発に変更しておきます"]


def test_no_terminal_stays_buffered_until_flush() -> None:
    agg = SentenceAggregator()
    assert agg.add("3日の京都出張のスケジュールなんですが") == []
    assert agg.pending() == "3日の京都出張のスケジュールなんですが"
    assert agg.flush() == ["3日の京都出張のスケジュールなんですが"]
    assert agg.pending() == ""


def test_pending_reflects_buffer_and_reset_clears_it() -> None:
    agg = SentenceAggregator()
    assert agg.add("部長") == []
    assert agg.pending() == "部長"
    agg.reset()
    assert agg.pending() == ""
    assert agg.flush() == []


def test_clean_complete_sentence_passes_through_unchanged() -> None:
    agg = SentenceAggregator()
    sentence = "今回こちらの新素材のものはいかがかと思いました。"
    assert agg.add(sentence) == [sentence]
    assert agg.pending() == ""


def test_multiple_sentences_in_one_add_are_all_returned() -> None:
    agg = SentenceAggregator()
    text = "この素材は吸水性に大変優れております。手触りもいいんです。"
    assert agg.add(text) == ["この素材は吸水性に大変優れております。", "手触りもいいんです。"]
    assert agg.pending() == ""


def test_terminal_followed_by_connective_forms_boundary() -> None:
    agg = SentenceAggregator()
    out = agg.add("北海道ですかそれはいいですねでは箱根の温泉に行きましょう")
    assert out == ["北海道ですか", "それはいいですね"]
    assert agg.flush() == ["では箱根の温泉に行きましょう"]


def test_multi_speaker_run_on_splits_without_clause_starter() -> None:
    # です followed directly by えっ (no recognized starter word) must still split.
    agg = SentenceAggregator()
    text = (
        "会議が午後一時からに変更になったそうなんですえっ一時間早まったの"
        "ええですから新幹線の時間も一時間早めた方がよろしいかと"
    )
    out = agg.add(text)
    assert out[0] == "会議が午後一時からに変更になったそうなんです", out
    assert "えっ一時間早まったの" in out, out


def test_continuation_particles_never_split() -> None:
    for text in (
        "部長三日の京都出張のスケジュールなんですが",
        "これは新幹線ですからお願いします",
        "明日ですって言っていました",
        "便利ですものね",
        "可能でしたらお願いします",
        "行けるかどうか確認します",
        "ご参加くださいましてありがとうございます",
    ):
        agg = SentenceAggregator()
        assert agg.add(text) == [], f"should not split: {text}"
        assert agg.pending() == text


def test_dangling_fragment_detection() -> None:
    agg = SentenceAggregator()
    for text in (
        "かと",
        "が",
        "ですね",
        "ますね",
        "ですよね",
        "だよね",
        "ですよ",
        "みたいな",
        "っぽい",
        "ということ",
        "ということで",
        "なんですけど",
        "なんですが",
    ):
        assert agg.is_dangling(text)
    assert not agg.is_dangling("はい")
    assert not agg.is_dangling("分かりました")


def test_polite_ne_not_orphaned_in_unpunctuated_runon() -> None:
    # ですね/ますね followed straight by the next clause must keep ね with its own
    # sentence, never orphan it onto the next ("です"+"ねとても…" was the bug).
    assert split_japanese_sentences("すぐに答える練習ですねとてもいいテーマです") == [
        "すぐに答える練習ですね",
        "とてもいいテーマです",
    ]
    # No content is lost regardless of where the boundary lands.
    src = "今日はいい天気ですねでも明日は雨です"
    assert "".join(split_japanese_sentences(src)) == src


def test_sentence_final_yo_does_not_corrupt_following_word() -> None:
    # よろしく after a greeting must stay intact when the greeting branch splits.
    assert split_japanese_sentences("おはようございますよろしくお願いします") == [
        "おはようございます",
        "よろしくお願いします",
    ]
    # A genuine sentence-final よ at end of buffer stays attached.
    assert split_japanese_sentences("いいですよ") == ["いいですよ"]


def test_polite_yo_ne_sequences_stay_whole() -> None:
    for text in (
        "今日はいい天気ですよね。",
        "そうですよね。",
        "明日行きますよね。",
    ):
        out = split_japanese_sentences(text)
        assert out == [text], out
        assert "".join(out) == text


def test_ascii_period_does_not_split_decimals_or_abbreviations() -> None:
    for text in (
        "価格は3.5ドルです",
        "U.S.A.について話します",
    ):
        out = split_japanese_sentences(text)
        assert out == [text], out
        assert "".join(out) == text


def test_greeting_followed_by_ga_continuation_stays_joined() -> None:
    text = "ありがとうございますが、始めます。"
    out = split_japanese_sentences(text)
    assert out == [text], out
    assert "".join(out) == text


def test_flush_merges_dangling_tail_with_previous_sentence() -> None:
    agg = SentenceAggregator()
    src = "今日は晴れです。が"
    agg._buffer = src
    out = agg.flush()
    assert out == [src], out
    assert "が" not in out
    assert "".join(out) == src
    assert agg.pending() == ""


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
