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
    # Additional business terms lacking proper-noun entries
    "交渉履歴": "lịch sử đàm phán",
    "交渉開始": "bắt đầu đàm phán",
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
    "バイタル": "Vital signs",
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
    "トークイベント": "sự kiện thảo luận",
    "トークイーブメント": "sự kiện thảo luận",
    "データベース": "database",
    "パイプライン": "pipeline",
    "マイグレーション": "migration",
    "モニタリング": "monitoring",
    "ロールバック": "rollback",
    "オンプレミス": "on-premises",
    "サーバーレス": "serverless",
    "アーキテクト": "architect",
    "スケジュール": "lịch trình",
    "ネットワーク": "network",
    "ライブラリ": "library",
    "バックログ": "backlog",
    "プロジェクト": "project",
    "スプリント": "sprint",
    "エンジニア": "engineer",
    "レビュー": "review",
    "スクラム": "Scrum",
    "アジェンダ": "chương trình nghị sự",
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
    # Overlaps with PROPER_NOUNS (kept separate in KATAKANA_TERMS because
    # pre-substitution runs before PROPER_NOUN_MAP, so the katakana value
    # must stand on its own — dead code warning: PROPER_NOUNS entries for
    # these terms are skipped during pre-processing)
    "クロステナント": "Cross-Tenant",
    "マルチテナント": "Multi-Tenant",
    "ダッシュボード": "dashboard",
    "ステータス": "trạng thái",
    "リリース": "release",
    "トリアージ": "triage",
    "バイタル": "vital signs",
    "インシデント": "incident",
}

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

if _OVERLAP_WARNINGS:
    import logging
    _log = logging.getLogger(__name__)
    for w in _OVERLAP_WARNINGS:
        _log.warning("Domain data inconsistency: %s", w)
