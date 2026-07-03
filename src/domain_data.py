"""Single source of truth for ALL domain terminology.

Every glossary/map in the pipeline derives from the CANONICAL dict defined here.
Adding a term to the canonical dict automatically propagates it to all consumers.

Three derived maps are built from the canonical dict, applied in this order:
  1. _KATAKANA_TERM_MAP (pre-processing): katakana loanwords → English/Vietnamese.
  2. _PROPER_NOUN_MAP (direct substitution): known kanji → Vietnamese/romaji,
     runs BEFORE glossary hints so overlapping keys are replaced cleanly.
  3. _BUSINESS_GLOSSARY (parenthetical hints): remaining kanji get (tiếng Việt)
     hints.  PROPER_NOUNS keys are excluded to prevent exact-key overlap.

NLLB_GLOSSARY in config.py is NOT derived here because NLLB needs English
substitutions (which it copies through), not Vietnamese.  Update it manually.

PHRASE_CORRECTIONS in post_correction.py is also independent — it handles ASR
misrecognition patterns (acoustic errors), not translation-domain mapping.
"""

from __future__ import annotations

# ── Canonical glossary ────────────────────────────────────────────────────
# Key:   Japanese term (kanji, katakana, or mixed)
# Value: Vietnamese translation (lowercase, standard orthography)
# One entry per term — NO DUPLICATES.  Add new terms here, not in the maps.
DOMAIN_TERMS: dict[str, str] = {
    # ── IT / business / project management ────────────────────────────
    "検討": "xem xét",
    "対応": "xử lý",
    "確認": "xác nhận",
    "共有": "chia sẻ",
    "実装": "triển khai",
    "移行": "chuyển đổi",
    "担当": "phụ trách",
    "従事": "tham gia",
    "活躍": "hoạt động",
    "導入": "đưa vào",
    "運用": "vận hành",
    "構築": "xây dựng",
    "開発": "phát triển",
    "設計": "thiết kế",
    "管理": "quản lý",
    "提案": "đề xuất",
    "説明": "giải thích",
    "報告": "báo cáo",
    "推進": "thúc đẩy",
    "連携": "liên kết",
    "基幹": "cơ bản/nền tảng",
    "案件": "hạng mục",
    "着手": "bắt đầu triển khai",
    "進捗": "tiến độ",
    "受け入れ": "tiếp nhận",
    "負荷試験": "kiểm tra tải",
    # ── Agile / dev ───────────────────────────────────────────────────
    "割り込み": "task gián đoạn",
    "割り込みタスク": "task gián đoạn",
    "親タスク": "parent task",
    "子タスク": "child task",
    # ── Emergency medical / rescue domain ─────────────────────────────
    "搬送": "vận chuyển",
    "引き継ぎ": "bàn giao",
    "出動": "điều động",
    "指令": "chỉ thị điều phối",
    "救急搬送": "vận chuyển cấp cứu",
    "現場": "hiện trường",
    "災害": "thảm họa",
    "患者": "bệnh nhân",
    "受入": "tiếp nhận",
    "救急": "cấp cứu",
    "救助": "cứu hộ",
    "救助活動": "hoạt động cứu hộ",
    "消火": "chữa cháy",
    "消火活動": "hoạt động chữa cháy",
    # ── Emergency activity statuses (from NeO-MATCH domain glossary) ──────
    "覚知": "tiếp nhận tin báo",
    "現着": "đến hiện trường",
    "接触": "tiếp xúc bệnh nhân",
    "車内収容": "đưa lên xe",
    "現発": "rời hiện trường",
    "病着": "đến bệnh viện",
    "引揚": "rút về sau bàn giao",
    "帰署": "về trụ sở",
    # ── Patient / inquiry status ───────────────────────────────────────────
    "受入可": "tiếp nhận được",
    "受入不可": "không tiếp nhận",
    "搬送完了": "hoàn tất vận chuyển",
    "照会": "hỏi tiếp nhận",
    "受入可否": "khả năng tiếp nhận",
    "応需設定": "cài đặt năng lực tiếp nhận",
}

# ── Proper nouns (direct substitution, NOT hints) ─────────────────────────
# These are organisations, places, and domain entities that should be directly
# replaced in the source text (not injected as parenthetical hints).
# They are EXCLUDED from _BUSINESS_GLOSSARY to prevent the overlap bug.
PROPER_NOUNS: dict[str, str] = {
    # IT companies
    "グーグル": "Google",
    "マイクロソフト": "Microsoft",
    "アップル": "Apple",
    "メタ": "Meta",
    "オラクル": "Oracle",
    # Japanese places
    "秋葉原": "Akihabara",
    "渋谷": "Shibuya",
    "新宿": "Shinjuku",
    "東京": "Tokyo",
    "大阪": "Osaka",
    "名古屋": "Nagoya",
    "福岡": "Fukuoka",
    "北海道": "Hokkaido",
    "六本木": "Roppongi",
    # Government / organisations
    "経済産業省": "Bộ Kinh tế Nhật",
    "デジタル庁": "Cơ quan Kỹ thuật số",
    "総務省": "Bộ Nội vụ Nhật",
    "東京消防庁": "Sở Cứu hỏa Tokyo",
    "消防庁": "Cục Cứu hỏa",
    "消防本部": "Sở chỉ huy Cứu hỏa",
    "消防団": "Đội Cứu hỏa",
    "消防署": "Trạm Cứu hỏa",
    "消防": "Cứu hỏa",
    "救急隊員": "Nhân viên cấp cứu",
    "医療機関スタッフ": "Nhân viên cơ sở y tế",
    "指令センター": "Trung tâm chỉ huy",
    "救急隊": "Đội Cấp cứu",
    "救急車": "Xe Cấp cứu",
    "警察": "Cảnh sát",
    "自衛隊": "Lực lượng Phòng vệ",
    "日本赤十字社": "Hội Chữ thập đỏ Nhật Bản",
    "DMAT": "DMAT",
    "医療機関": "Cơ sở y tế",
    "病院": "Bệnh viện",
    "EMIS": "EMIS",
    "広域災害": "Thảm họa diện rộng",
    "広域地図": "Bản đồ diện rộng",
    # Emergency / rescue domain entities (also have entries in DOMAIN_TERMS
    # — the PROPER_NOUNS version wins because it's applied FIRST as direct
    # substitution; see _translate_one  proper-nouns-before-glossary order)
    "傷病者": "Nạn nhân",
    "搬送先": "Nơi tiếp nhận",
    "搬送者": "Người vận chuyển",
    "搬送元": "Nơi xuất phát vận chuyển",
    "搬送決定": "Quyết định vận chuyển",
    "多数傷病者": "Nạn nhân hàng loạt",
    "病院連携": "Liên kết bệnh viện",
    "通信指令": "Điều phối thông tin",
    "通信指令台": "Tổng đài điều phối",
    "交渉": "Đàm phán",
    "交渉状態": "Trạng thái đàm phán",
    "交渉履歴": "Lịch sử đàm phán",
    "交渉開始": "Bắt đầu đàm phán",
    "現場到着": "Đến hiện trường",
    "災害時": "Khi có thảm họa",
    "トリアージ": "Triage",
    "トリアージタグ": "Triage tag",
    "ホットゾーン": "Hot zone",
    "ストレッチャー": "Stretcher",
    "レスキュー": "Rescue",
    "パラメディック": "Paramedic",
    "メディカルコントロール": "Medical control",
    "インシデントコマンド": "Incident command",
    "ポッドキャスト": "Podcast",
    # Extra IT terms
    "クロステナント": "Cross-Tenant",
    "マルチテナント": "Multi-Tenant",
    "インシデント": "Incident",
    "ダッシュボード": "Dashboard",
    "レビュー": "Review",
    "リリース": "Release",
}

# ── Katakana loanwords (pre-processed before LLM) ────────────────────────
# Convert Japanese katakana to English/Vietnamese equivalents so the LLM
# doesn't have to guess katakana readings.  These are applied as direct
# substitutions (not hints) BEFORE any other processing.
KATAKANA_TERMS: dict[str, str] = {
    "ソリューションアーキテクト": "Solution Architect",
    "ミッションクリティカル": "mission-critical",
    "プレゼンテーション": "presentation",
    "インテグレーション": "integration",
    "パブリックセクター": "public sector",
    "プルリクエスト": "pull request",
    "フレームワーク": "framework",
    "マイクロサービス": "microservice",
    "リファインメント": "refinement",
    "リファクタリング": "refactoring",
    "インターフェース": "interface",
    "ステークホルダー": "stakeholder",
    "ロードバランサー": "load balancer",
    "クラウドファースト": "Cloud First",
    "エンタープライズ": "enterprise",
    "ウェブサービス": "Web Services",
    "ユニットテスト": "unit test",
    "コンポーネント": "component",
    "フィードバック": "feedback",
    "マイルストーン": "milestone",
    "イーシーツー": "EC2",
    "エスキューエス": "SQS",
    "テクノロジー": "Technology",
    "トークイベント": "talk event",
    "トークイーブメント": "talk event",
    "データベース": "database",
    "パイプライン": "pipeline",
    "マイグレーション": "migration",
    "モニタリング": "monitoring",
    "ロールバック": "rollback",
    "オンプレミス": "on-premises",
    "サーバーレス": "serverless",
    "アーキテクト": "architect",
    "スケジュール": "schedule",
    "ネットワーク": "network",
    "ライブラリ": "library",
    "バックログ": "backlog",
    "プロジェクト": "project",
    "スプリント": "sprint",
    "エンジニア": "engineer",
    "レビュー": "review",
    "スクラム": "Scrum",
    "アジェンダ": "agenda",
    "アマゾン": "Amazon",
    "インフラ": "infrastructure",
    "オンプレ": "on-premises",
    "クラウド": "Cloud",
    "コミット": "commit",
    "コンテナ": "container",
    "サーバー": "server",
    "デバッグ": "debug",
    "デプロイ": "deploy",
    "ブランチ": "branch",
    "モジュール": "module",
    "リリース": "release",
    "ロボット": "robot",
    "キャッシュ": "cache",
    "エスサン": "S3",
    "マージ": "merge",
    "ラムダ": "Lambda",
    "タスク": "task",
    # Emergency medical / rescue katakana
    "バイタル": "vital signs",
    "トリアージ": "triage",
    "ストレッチャー": "stretcher",
    "インシデント": "incident",
    "コマンド": "command",
    "コントロール": "control",
    "オペレーション": "operation",
    "ディスパッチ": "dispatch",
    "プロトコル": "protocol",
    "トリアージタグ": "triage tag",
    "ホットゾーン": "hot zone",
    "レスキュー": "rescue",
    "パラメディック": "paramedic",
    "メディカル": "medical",
    "エマージェンシー": "emergency",
    "マニュアル": "manual",
    "シミュレーション": "simulation",
    "トレーニング": "training",
    # Additional IT katakana
    "パーミッション": "permission",
    "ロールベース": "role-based",
    "クエリ": "query",
    "レコード": "record",
    "バックエンド": "backend",
    "フロントエンド": "frontend",
    "ユーザーストーリー": "user story",
    "エンドポイント": "endpoint",
    "トークン": "token",
    "セッション": "session",
    "ポリシー": "policy",
    "プロビジョニング": "provisioning",
    "オーケストレーション": "orchestration",
    "コンフィグ": "config",
    "ペイロード": "payload",
    "レスポンス": "response",
    "リクエスト": "request",
    "ステータスコード": "status code",
    "アクセス権": "access rights",
    "サイロ": "silo",
    "カラム": "column",
    "スキーマ": "schema",
    # IT katakana from 2026-06-24 meeting (kept English so they stay consistent)
    "フィルター": "filter",
    "フィルタリング": "filtering",
    "ドキュメント": "document",
    "グループ": "group",
    "ウォーターフォール": "waterfall",
    "フェーズ": "phase",
    "メーター": "metrics",
}

# ── Filler / back-channel utterances (shared by router + LLM backends) ────
# Pure back-channels, greetings, and fixed acknowledgements whose Vietnamese
# rendering is invariant. Both backends short-circuit these to skip a network
# round-trip (router) / LLM inference (llm) and to stop weak models from
# over-expanding them ("はい" → "Vâng được ạ"). Single source of truth so the
# two backends never drift (they had 22 vs 72 entries before consolidation).
# NOTE: entries like 聞こえますか/難しいですか are fixed meeting phrases with one
# correct rendering, not context-dependent questions — safe to canned-answer.
FILLER_MAP: dict[str, str] = {
    # ── Back-channels / acknowledgements ──────────────────────────────
    "うん": "Vâng", "うんうん": "Vâng, vâng", "うんうんはい": "Vâng, vâng, đúng rồi",
    "うんうんうん": "Vâng vâng vâng", "はい": "Vâng", "はいはい": "Vâng, vâng",
    "はいはいはい": "Vâng vâng vâng", "はいもね": "Vâng, đúng nhỉ", "ええ": "Vâng",
    "え": "Ơ", "えっと": "À...", "あの": "À...", "あのう": "À...", "ああ": "À",
    "あ": "À", "まあ": "Thôi thì", "なるほど": "Ra vậy", "なるほどね": "Ra vậy nhỉ",
    "そうですね": "Đúng vậy nhỉ", "そうそう": "Đúng, đúng", "そうそうそう": "Đúng đúng đúng",
    "ですね": "Đúng vậy", "っていう": "Nghĩa là", "ます": "...", "ねえ": "Này",
    "ねえねえ": "Này này", "うんねえねえ": "Vâng, này này",
    "そっか": "Vậy à", "そうか": "Vậy à", "そうかそうか": "Vậy à, vậy à",
    # ── Greetings / meeting phrases ───────────────────────────────────
    "こんにちは": "Xin chào", "こんばんは": "Chào buổi tối",
    "おはようございます": "Chào buổi sáng", "お願いします": "Xin hãy giúp",
    "お願いいたします": "Xin vui lòng", "お世話になっております": "Cảm ơn đã luôn giúp đỡ",
    "お世話になります": "Xin được nhờ vả", "お疲れ様です": "Xin chào",
    "お疲れ様でした": "Cảm ơn đã vất vả", "よろしくお願いします": "Xin vui lòng hỗ trợ",
    "よろしくお願いいたします": "Rất mong được hỗ trợ", "ありがとうございます": "Cảm ơn",
    "ありがとうございました": "Cảm ơn rất nhiều",
    "先日は打ち合わせありがとうございました": "Cảm ơn về cuộc họp hôm trước",
    "先日はありがとうございました": "Cảm ơn về hôm trước",
    "いえいえこちらこそ": "Không không, bên tôi mới phải cảm ơn",
    "いかがですか": "Thế nào ạ?", "いかがでしょうか": "Thế nào ạ?",
    "難しいですか": "Có khó không?", "難しいと思います": "Tôi nghĩ là khó",
    "すみません": "Xin lỗi", "申し訳ございません": "Thành thật xin lỗi",
    "失礼します": "Xin phép", "失礼いたします": "Xin phép ạ",
    "承知しました": "Tôi đã hiểu", "了解です": "Đã hiểu", "了解しました": "Đã hiểu rồi",
    "かしこまりました": "Vâng, tôi hiểu", "おっしゃる通りです": "Đúng như bạn nói",
    "そういうことですね": "À ra là vậy", "その通りです": "Đúng vậy",
    "間違いないです": "Không sai", "以上です": "Trên đây là tất cả",
    "以上になります": "Trên đây là tất cả", "みなさんこんにちは": "Xin chào mọi người",
    "皆さんこんにちは": "Xin chào mọi người", "では始めましょう": "Vậy chúng ta bắt đầu nhé",
    "始めましょう": "Bắt đầu thôi", "それでは": "Vậy thì",
    "ちょっと待ってください": "Xin đợi một chút", "少々お待ちください": "Xin vui lòng chờ một chút",
    "聞こえますか": "Nghe được không?", "見えますか": "Nhìn thấy không?",
    "大丈夫です": "Không sao", "問題ないです": "Không có vấn đề gì",
}


def match_filler(cleaned: str) -> str | None:
    """Return the canned Vietnamese for a filler/back-channel, or None.

    Exact match first, then a prefix match for a filler the ASR truncated by ONE
    trailing mora (e.g. ``ありがとうございま`` for ``ありがとうございます``). Two
    guards keep real words from false-matching a longer filler:
      - input must be >=3 chars — a 1-2 char string (は particle, です copula,
        なる verb) is real content, not a cut filler; matching です→ですね→"Đúng
        vậy" or なる→なるほど→"Ra vậy" is a mistranslation.
      - at most 1 char may be missing — ASR drops the final mora (す/した→し),
        not two; allowing a 2-char gap let です match ですね and なる match なるほど.
    Shared by both backends so their short-circuit behaviour never drifts.
    """
    result = FILLER_MAP.get(cleaned)
    if result is None and len(cleaned) >= 3:
        for key, val in FILLER_MAP.items():
            if key.startswith(cleaned) and len(cleaned) >= len(key) - 1:
                return val
    return result


# ── Hard refusal markers (shared by router + LLM backends) ────────────────
# Unambiguous "I won't/can't translate" phrases. A chat model that refuses
# instead of translating must have its output rejected, not shown as a subtitle
# or (worse) fed into the context window where it poisons following segments.
# Kept deliberately SHORT and unambiguous — softer preamble phrases ("tôi hiểu
# rồi") are NOT here because they also occur in valid translations. Substring
# match, case-insensitive.
HARD_REFUSAL_PATTERNS: tuple[str, ...] = (
    "tôi sẽ không dịch",
    "tôi không thể dịch",
    "tôi xin lỗi",
    "nội dung nhạy cảm",
    "không phù hợp",
    "i cannot translate",
    "i won't translate",
    "i can't translate",
)

# ── Build-time validation ────────────────────────────────────────────────
# Ensure no term appears in both DOMAIN_TERMS and PROPER_NOUNS with a
# DIFFERENT Vietnamese translation (capitalisation aside, the meaning must
# be the same).
_OVERLAP_WARNINGS: list[str] = []
for term, vi in DOMAIN_TERMS.items():
    if term in PROPER_NOUNS:
        pn_vi = PROPER_NOUNS[term]
        if vi.lower() != pn_vi.lower():
            _OVERLAP_WARNINGS.append(
                f"DOMAIN_TERMS['{term}']='{vi}' vs PROPER_NOUNS['{term}']='{pn_vi}'"
            )
# KATAKANA_TERMS is applied before PROPER_NOUNS, so a PROPER_NOUNS entry for the
# same key is shadowed (dead). Case differences are harmless (both romanize the
# same), but a genuine meaning divergence would be a silent bug — flag it.
for term, kt_vi in KATAKANA_TERMS.items():
    if term in PROPER_NOUNS and kt_vi.lower() != PROPER_NOUNS[term].lower():
        _OVERLAP_WARNINGS.append(
            f"KATAKANA_TERMS['{term}']='{kt_vi}' vs PROPER_NOUNS['{term}']='{PROPER_NOUNS[term]}'"
        )

if _OVERLAP_WARNINGS:
    import logging
    _log = logging.getLogger(__name__)
    for w in _OVERLAP_WARNINGS:
        _log.warning("Domain data inconsistency: %s", w)
