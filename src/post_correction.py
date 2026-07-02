"""Post-ASR text correction for persistent misrecognitions.

Hotwords cannot fix all errors — e.g., the streaming path uses greedy_search
(zero hotword effect), and some confusions involve phonetically dissimilar words
where the acoustic model is simply wrong (テナント→祖母).

This module applies deterministic string replacements AFTER ASR produces text,
providing a safety net for domain-specific terms in the rescue/dispatch context.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─── Domain-specific corrections ─────────────────────────────────────────────
# Format: wrong_text → correct_text
# Only add entries verified from real meeting logs (evidence/run_*.jsonl).
# Order matters: longer patterns checked first to avoid partial matches.
PHRASE_CORRECTIONS: dict[str, str] = {
    # ASR confuses テナント (tenanto) with 祖母 (sobo) — zero phonetic overlap
    "クロス祖母": "クロステナント",
    "マルチ祖母": "マルチテナント",
    # 交渉 (kōshō) vs 口臭 (kōshū) — shared prefix こう
    "口臭状態": "交渉状態",
    "口臭履歴": "交渉履歴",
    "口臭開始": "交渉開始",
    "口臭": "交渉",
    # 消防 (shōbō) vs 昭和 (shōwa) — shared prefix しょう
    # Only replace when NOT followed by era-related context (年, 時代)
    "昭和テナント": "消防テナント",
    "昭和署": "消防署",
    # パーミッション missing long vowel
    "パミッション": "パーミッション",
    # テスト環境 mishearing
    "マスト環境": "テスト環境",
    # dev2 loanword
    "デブ二": "dev2",
    "デブツー": "dev2",
    # ユースケース dropped prefix — only when NOT already preceded by ユー
    "ペナント": "テナント",
    # ASR misrecognitions from meeting evidence (phonetically similar)
    "海水エコ": "解消",
    "別居": "別件",
    "着水": "着手",
    "逆襲": "着手",
    "非性能": "非機能",
    "相場": "案件",
    "イクライド": "Eclipse",
    "卒業中": "調整中",
    # ASR confusions from 2026-06-08 rescue dispatch meeting
    "感染していて": "案件でいって",
    "大森アート": "大森の方",
    "化け物": "確認もの",
    # ASR misrecognitions from 2026-06-09 meeting
    "風のタスク": "フォローのタスク",
    "断作業": "タスク",
    "ウェアタスク": "親タスク",
    "ビント": "スプリント",
    "ビデー": "API",
    "寒い週": "案件週",
    # ASR misrecognitions from 2026-06-24 meeting (katakana mishearings)
    # Evidence: speaker said filter/parent-task but ASR emitted these forms.
    "フィリット": "フィルター",      # "filter" misheard
    "フィリッと": "フィルター",
    "マタスク": "親タスク",          # "parent task" (oya-task) misheard
    "ウェブフィルター": "Webフィルター",
    # Acronyms spoken as katakana → canonical Latin form (matches domain_data).
    "ディーマット": "DMAT",          # emergency medical assistance team
    "イーミス": "EMIS",              # emergency medical info system
}

# Sorted longest-first once at import time — post_correct() is on the hot path.
# ponytail: module-level sort; rebuild if PHRASE_CORRECTIONS grows past ~200 entries and profiling shows overhead
_SORTED_CORRECTIONS: list[tuple[str, str]] = sorted(
    PHRASE_CORRECTIONS.items(), key=lambda kv: -len(kv[0])
)
assert all(
    len(_SORTED_CORRECTIONS[i][0]) >= len(_SORTED_CORRECTIONS[i + 1][0])
    for i in range(len(_SORTED_CORRECTIONS) - 1)
), "BUG: _SORTED_CORRECTIONS is not longest-first"

# Context-aware corrections: only apply when surrounding text matches a pattern
CONTEXT_CORRECTIONS: list[tuple[re.Pattern, str, str]] = [
    # スケース → ユースケース only when NOT preceded by ユー
    (re.compile(r"(?<!ユー)スケース"), "スケース", "ユースケース"),
    # 祖母 → 消防 when in dispatch/fire context (followed by system terms)
    (re.compile(r"祖母(?=の|署|隊|車|局|側|が)"), "祖母", "消防"),
    # 昭和 → 消防 when NOT followed by era markers
    (re.compile(r"昭和(?!年|時代|[0-9０-９])"), "昭和", "消防"),
    # 浮世 (ukiyo) → 設計 (sekkei) when 設計 appears in preceding text
    (re.compile(r"(?<=設計の件ですけど)恐らく浮世"), "恐らく浮世", "恐らく設計"),
]


def post_correct(text: str) -> str:
    """Apply domain-specific corrections to ASR output text.

    Returns corrected text. Safe to call on any input — returns unchanged
    text if no corrections apply.
    """
    if not text:
        return text

    original = text

    # Phase 1: exact phrase replacements (longest-first, pre-sorted at import)
    for wrong, right in _SORTED_CORRECTIONS:
        if wrong in text:
            text = text.replace(wrong, right)

    # Phase 2: context-aware regex replacements
    for pattern, _, replacement in CONTEXT_CORRECTIONS:
        text = pattern.sub(replacement, text)

    if text != original:
        logger.debug("Post-correction: '%s' → '%s'", original, text)

    return text

