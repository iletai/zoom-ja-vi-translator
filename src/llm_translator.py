"""LLM-based Japanese-to-Vietnamese translation via llama-cpp-python."""
from __future__ import annotations

import inspect
import logging
import pathlib
import re
import threading
import unicodedata
from collections import deque
from typing import Any

import config
from src.sentence_aggregator import split_japanese_sentences

try:
    from src.translator import join_translations
except Exception:  # pragma: no cover - fallback for partial environments
    def join_translations(sources, translations):
        """Positionally join VI ``translations`` for their JP ``sources``."""
        parts: list[str] = []
        dropped: list[tuple[int, str]] = []
        for idx, (src, vi) in enumerate(zip(sources, translations)):
            if vi and vi.strip():
                parts.append(vi.strip())
            elif src and src.strip():
                parts.append("(...)")
                dropped.append((idx, src))
        return " ".join(parts), dropped


logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Bạn là máy dịch Nhật→Việt chuyên IT. "
    "Nhận tiếng Nhật, xuất ĐÚNG MỘT DÒNG tiếng Việt, không thêm gì khác. "
    "Giữ nguyên: Cloud, AWS, API, deploy, sprint, IoT, AI, EC2, S3, Lambda. các thuật ngữ trong ngành IT."
    "Tên riêng giữ romaji: Tokyo, Shibuya, Akihabara."
    "Bạn tuyệt đối phải trả lời với độ chính xác cao, tin cậy tránh phản hồi sai xót ngoài ngôn ngữ được định nghĩa."
)


class LlmTranslator:
    """Drop-in Japanese-to-Vietnamese translator backed by llama.cpp."""

    _FEW_SHOT_EXAMPLES = (
        ("JA: 次のsprintでAPIを修正します", "VI: Chúng tôi sẽ sửa API trong sprint tới."),
    )
    _MAX_PROMPT_CONTEXT_SENTENCES = 1

    def __init__(self) -> None:
        model_path = pathlib.Path(str(getattr(config, "LLM_MODEL_PATH", ""))).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"LLM GGUF model file not found: {model_path}. "
                "Set config.LLM_MODEL_PATH to a valid GGUF file path."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        self.model_path = model_path
        self.system_prompt = getattr(config, "LLM_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)
        self.n_ctx = max(128, int(getattr(config, "LLM_N_CTX", 1024)))
        self.n_threads = max(
            1,
            int(getattr(config, "LLM_N_THREADS", getattr(config, "_PHYSICAL_CORES", 1) or 1)),
        )
        self.n_batch = max(1, int(getattr(config, "LLM_N_BATCH", 1024)))
        self.n_gpu_layers = int(getattr(config, "LLM_N_GPU_LAYERS", -1))
        self.temperature = float(getattr(config, "LLM_TEMPERATURE", 0.1))
        self.top_p = float(getattr(config, "LLM_TOP_P", 0.3))
        self.frequency_penalty = float(getattr(config, "LLM_FREQUENCY_PENALTY", 0.1))
        self.max_tokens = max(1, int(getattr(config, "LLM_MAX_TOKENS", 150)))
        self.context_sentences = max(0, int(getattr(config, "LLM_CONTEXT_SENTENCES", 3)))
        self._keep_context = self.context_sentences > 0
        self._history: deque[tuple[str, str]] = deque(
            maxlen=self.context_sentences if self._keep_context else 1
        )
        self._lock = threading.Lock()

        llama_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "use_mlock": bool(getattr(config, "LLM_USE_MLOCK", False)),
            "verbose": False,
        }
        try:
            llama_params = inspect.signature(Llama.__init__).parameters
        except (TypeError, ValueError):
            llama_params = {}
        if "flash_attn" in llama_params:
            llama_kwargs["flash_attn"] = True

        self.llm = Llama(**llama_kwargs)

        try:
            self.warmup()
        except Exception as exc:  # pragma: no cover - best-effort latency optimization
            logger.warning("LLM translator warmup failed: %s", exc)

    def translate(self, text: str) -> str:
        """Translate Japanese ``text`` to Vietnamese without raising on failures."""
        return self._translate_text(text)

    def translate_many(self, texts: list[str]) -> list[str]:
        """Translate ``texts`` one-by-one so later items benefit from prior context."""
        return [self._translate_text(text) for text in texts]

    def warmup(self) -> None:
        """Run one tiny translation to avoid first-call lag."""
        self._translate_one("テスト", update_context=False)

    def _translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        try:
            if getattr(config, "TRANSLATE_SPLIT_SENTENCES", True):
                sentences = split_japanese_sentences(text)
                if len(sentences) > 1:
                    translations = [self._translate_one(sentence) for sentence in sentences]
                    joined, dropped = join_translations(sentences, translations)
                    for idx, src in dropped:
                        logger.warning("Empty VI translation for JP sentence[%d]: %r", idx, src)
                    return joined
                if len(sentences) == 1:
                    text = sentences[0]
            return self._translate_one(text)
        except Exception as exc:
            logger.warning("LLM translation failed for %r: %s", text, exc)
            return ""

    # Hardcoded filler word translations (LLM struggles with these)
    _FILLER_MAP = {
        "うん": "Vâng",
        "うんうん": "Vâng, vâng",
        "うんうんはい": "Vâng, vâng, đúng rồi",
        "はい": "Vâng",
        "はいはい": "Vâng, vâng",
        "ええ": "Vâng",
        "え": "Ơ",
        "えっと": "À...",
        "あの": "À...",
        "ああ": "À",
        "あ": "À",
        "まあ": "Thôi thì",
        "なるほど": "Ra vậy",
        "そうですね": "Đúng vậy nhỉ",
        "そうそう": "Đúng, đúng",
        "ですね": "Đúng vậy",
        "っていう": "Nghĩa là",
        "ます": "...",
        # ─── Common meeting greetings/phrases (bypass LLM for reliability) ───
        "こんにちは": "Xin chào",
        "こんばんは": "Chào buổi tối",
        "おはようございます": "Chào buổi sáng",
        "お願いします": "Xin hãy giúp",
        "お願いいたします": "Xin vui lòng",
        "お世話になっております": "Cảm ơn đã luôn giúp đỡ",
        "お世話になります": "Xin được nhờ vả",
        "お疲れ様です": "Xin chào",
        "お疲れ様でした": "Cảm ơn đã vất vả",
        "よろしくお願いします": "Xin vui lòng hỗ trợ",
        "よろしくお願いいたします": "Rất mong được hỗ trợ",
        "ありがとうございます": "Cảm ơn",
        "ありがとうございました": "Cảm ơn rất nhiều",
        "先日は打ち合わせありがとうございました": "Cảm ơn về cuộc họp hôm trước",
        "先日はありがとうございました": "Cảm ơn về hôm trước",
        "いえいえこちらこそ": "Không không, bên tôi mới phải cảm ơn",
        "いかがですか": "Thế nào ạ?",
        "いかがでしょうか": "Thế nào ạ?",
        "難しいですか": "Có khó không?",
        "難しいと思います": "Tôi nghĩ là khó",
        "すみません": "Xin lỗi",
        "申し訳ございません": "Thành thật xin lỗi",
        "失礼します": "Xin phép",
        "失礼いたします": "Xin phép ạ",
        "承知しました": "Tôi đã hiểu",
        "了解です": "Đã hiểu",
        "了解しました": "Đã hiểu rồi",
        "かしこまりました": "Vâng, tôi hiểu",
        "おっしゃる通りです": "Đúng như bạn nói",
        "そういうことですね": "À ra là vậy",
        "その通りです": "Đúng vậy",
        "間違いないです": "Không sai",
        "以上です": "Trên đây là tất cả",
        "以上になります": "Trên đây là tất cả",
        "みなさんこんにちは": "Xin chào mọi người",
        "皆さんこんにちは": "Xin chào mọi người",
        "では始めましょう": "Vậy chúng ta bắt đầu nhé",
        "始めましょう": "Bắt đầu thôi",
        "それでは": "Vậy thì",
        "ちょっと待ってください": "Xin đợi một chút",
        "少々お待ちください": "Xin vui lòng chờ một chút",
        "聞こえますか": "Nghe được không?",
        "見えますか": "Nhìn thấy không?",
        "大丈夫です": "Không sao",
        "問題ないです": "Không có vấn đề gì",
    }

    # Keigo simplification: verbose honorific → plain form (pre-processing).
    _KEIGO_SIMPLIFY = [
        ("させていただきます", "します"),
        ("させていただく", "する"),
        ("させていただいて", "して"),
        ("させていただければ", "すれば"),
        ("いただけますでしょうか", "もらえますか"),
        ("いただけますか", "もらえますか"),
        ("いただきたい", "ほしい"),
        ("でございます", "です"),
        ("申し上げます", "言います"),
        ("おっしゃる", "言う"),
        ("いらっしゃる", "いる"),
        ("ございます", "あります"),
        ("存じます", "思います"),
        ("いたします", "します"),
        ("いたしました", "しました"),
    ]

    # Katakana IT terms → English. Used as pre-processing substitution (mid-sentence).
    # Sorted by length (longest first) at substitution time to avoid partial matches.
    _KATAKANA_TERM_MAP = {
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
    }

    _BUSINESS_GLOSSARY = {
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
    }

    # Proper nouns (place names, companies) that the model fails to transliterate.
    # Pre-processed to romaji/English before LLM translation.
    _PROPER_NOUN_MAP = {
        # Japanese cities/areas
        "秋葉原": "Akihabara",
        "渋谷": "Shibuya",
        "新宿": "Shinjuku",
        "東京": "Tokyo",
        "大阪": "Osaka",
        "名古屋": "Nagoya",
        "福岡": "Fukuoka",
        "北海道": "Hokkaido",
        "六本木": "Roppongi",
        # IT Companies
        "グーグル": "Google",
        "マイクロソフト": "Microsoft",
        "アップル": "Apple",
        "メタ": "Meta",
        "オラクル": "Oracle",
        # Organizations
        "経済産業省": "Bộ Kinh tế Nhật",
        "デジタル庁": "Cơ quan Kỹ thuật số",
        "総務省": "Bộ Nội vụ Nhật",
    }

    # CJK single-digit numerals → Arabic digit (applied before kanji stripping).
    # Compound markers (十百千万億) are stripped separately — they can't be
    # trivially converted without full kanji number parsing.
    _CJK_NUMERAL_MAP = str.maketrans(
        {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
         "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
         "十": None, "百": None, "千": None, "万": None, "億": None}
    )

    # Vietnamese diacritical characters
    _VI_DIACRITICS_RE = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', re.IGNORECASE)
    _ASCII_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    _COMMON_ENGLISH_WORDS = {
        "a", "an", "and", "are", "because", "but", "difficult", "everyone",
        "had", "has", "have", "hello", "however", "is", "it", "its", "may",
        "might", "no", "not", "or", "that", "the", "this", "was", "were",
        "will", "would", "yes", "could", "should", "can",
    }
    _LEADING_ENGLISH = {
        "unfortunately", "however", "but", "well", "so", "yes", "no",
        "okay", "actually", "basically", "honestly", "also", "and",
    }
    _UOC_BIAS_PREFIXES = (
        "Ước mơ là ",
        "Ước mơ ",
        "Ước mong là ",
        "Ước mong ",
        "Ước gì ",
    )
    _LITERAL_UOC_SOURCE_HINTS = ("夢", "ゆめ", "願望", "願う", "願って", "祈り", "祈る")

    def _translate_one(self, text: str, update_context: bool = True) -> str:
        cleaned = text.strip() if text else ""
        if not cleaned:
            return ""

        # Check filler words first (no LLM needed)
        filler_result = self._FILLER_MAP.get(cleaned)
        # Prefix match for truncated fillers (ASR sometimes cuts final す/した)
        if not filler_result:
            for filler_key, filler_val in self._FILLER_MAP.items():
                if filler_key.startswith(cleaned) and len(cleaned) >= len(filler_key) - 2:
                    filler_result = filler_val
                    break
        if filler_result:
            if update_context and self._keep_context:
                self._history.append((cleaned, filler_result))
            return filler_result

        # Check if entire input is a single katakana IT term
        katakana_result = self._KATAKANA_TERM_MAP.get(cleaned)
        if katakana_result:
            if update_context and self._keep_context:
                self._history.append((cleaned, katakana_result))
            return katakana_result

        # Simplify keigo (honorific) patterns to plain form for cleaner translation.
        processed = cleaned
        for keigo, plain in self._KEIGO_SIMPLIFY:
            processed = processed.replace(keigo, plain)

        # Pre-process: replace known katakana IT terms with English equivalents
        # so the LLM doesn't mistranslate them. Longest match first.
        for ja_term, en_term in sorted(
            self._KATAKANA_TERM_MAP.items(), key=lambda x: -len(x[0])
        ):
            processed = processed.replace(ja_term, en_term)

        # Inject Vietnamese hints for kanji business terms (helps LLM accuracy)
        for kanji, hint in self._BUSINESS_GLOSSARY.items():
            if kanji in processed:
                processed = processed.replace(kanji, f"{kanji}({hint})", 1)

        # Replace proper nouns with romaji/English equivalents
        for noun, replacement in self._PROPER_NOUN_MAP.items():
            processed = processed.replace(noun, replacement)

        try:
            with self._lock:
                raw_prompt = self._build_raw_prompt(processed)
                dynamic_max_tokens = min(
                    self.max_tokens,
                    max(50, len(processed) * 3),
                )
                response = self.llm.create_completion(
                    prompt=raw_prompt,
                    max_tokens=dynamic_max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    repeat_penalty=1.1,
                    stop=["\n", "<|im_end|>", "JA:", "JP:", "\nJA:", " Here ", " This is", "I am", "I will", "Let me",
                          "的", "是", "了", "在", "不", "我们"],
                )
                raw_text = response["choices"][0]["text"] if response.get("choices") else ""
                logger.debug("LLM raw output for %r: %r", cleaned[:40], raw_text[:120])
                translation = self._fix_uoc_bias(cleaned, self._clean_translation(raw_text))
                # Reject if translation is absurdly long relative to source.
                # Short Japanese inputs can legitimately expand much more in Vietnamese.
                if translation and len(translation) > max(120, len(cleaned) * 6):
                    logger.warning(
                        "LLM output too long vs source (%d vs %d chars), rejecting: %r",
                        len(translation), len(cleaned), translation[:80],
                    )
                    translation = ""
                if not translation:
                    is_fragment = self._is_incomplete_fragment(cleaned)
                    if is_fragment:
                        # Incomplete fragment: use a prompt that explicitly
                        # instructs the model to translate the partial meaning.
                        retry_prompt = (
                            "<|im_start|>system\n"
                            "Máy dịch Nhật→Việt. Câu nhập có thể chưa hoàn chỉnh (bị cắt giữa chừng). "
                            "Hãy dịch phần nghĩa đã có, thêm '...' ở cuối nếu câu chưa kết thúc. "
                            "Chỉ xuất bản dịch tiếng Việt. KHÔNG dùng tiếng Trung/Hàn/Thái.<|im_end|>\n"
                            "<|im_start|>user\n"
                            f"JA: {processed}<|im_end|>\n"
                            "<|im_start|>assistant\n"
                            "VI: "
                        )
                    else:
                        retry_prompt = (
                            "<|im_start|>system\n"
                            "Máy dịch Nhật→Việt. Chỉ xuất bản dịch tiếng Việt. "
                            "TUYỆT ĐỐI KHÔNG dùng tiếng Trung.<|im_end|>\n"
                            "<|im_start|>user\n"
                            f"JA: {processed}<|im_end|>\n"
                            "<|im_start|>assistant\n"
                            "VI: "
                        )
                    retry_response = self.llm.create_completion(
                        prompt=retry_prompt,
                        max_tokens=dynamic_max_tokens,
                        temperature=0.3,
                        top_p=0.5,
                        frequency_penalty=0.2,
                        repeat_penalty=1.2,
                        stop=["<|im_end|>", "\n\n", "JA:", "JP:", " Here ", " This is", "I am", "I will", "Let me",
                              "的", "是", "了", "在", "不", "我们"],
                    )
                    raw_retry = (
                        retry_response["choices"][0]["text"]
                        if retry_response.get("choices")
                        else ""
                    )
                    translation = self._fix_uoc_bias(cleaned, self._clean_translation(raw_retry))
                    # Length ratio check for retry too
                    if translation and len(translation) > max(120, len(cleaned) * 6):
                        logger.warning(
                            "Retry output too long vs source (%d vs %d chars): %r",
                            len(translation), len(cleaned), translation[:80],
                        )
                        translation = ""
                    if not translation:
                        logger.warning(
                            "Empty LLM translation for JP sentence (after retry): %r",
                            cleaned,
                        )
                        return ""
                    logger.info("Retry succeeded for: %r -> %r", cleaned, translation[:50])
                # Only add valid translations to context (prevents pollution)
                if update_context and self._keep_context and len(translation) < 300:
                    self._history.append((cleaned, translation))
                return translation
        except Exception as exc:
            logger.warning("LLM translation failed for %r: %s", cleaned, exc)
            return ""

    def _build_raw_prompt(self, text: str) -> str:
        """Build raw ChatML prompt with TRUE assistant prefill."""
        parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"]
        few_shots = getattr(self, "_FEW_SHOT_EXAMPLES", ())

        # Estimate token budget: reserve space for output within n_ctx.
        # Rough heuristic: 1 token ≈ 3 chars for mixed CJK/Latin text.
        max_output_tokens = min(self.max_tokens, max(50, len(text) * 3))
        token_budget = self.n_ctx - max_output_tokens
        used_tokens = len(self.system_prompt) // 3 + 10

        for user_ex, asst_ex in few_shots:
            cost = (len(user_ex) + len(asst_ex)) // 3 + 12
            if used_tokens + cost > token_budget - 80:
                break
            parts.append(f"<|im_start|>user\n{user_ex}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{asst_ex}<|im_end|>\n")
            used_tokens += cost

        prompt_context_sentences = min(
            getattr(self, "context_sentences", 0),
            getattr(self, "_MAX_PROMPT_CONTEXT_SENTENCES", 1),
        )
        if self._keep_context and self._history and len(text) > 4 and prompt_context_sentences > 0:
            recent = list(self._history)[-prompt_context_sentences:]
            for jp, vi in recent:
                part_cost = (len(jp) + len(vi)) // 3 + 12
                if used_tokens + part_cost > token_budget - 40:
                    break
                parts.append(f"<|im_start|>user\nJA: {jp}<|im_end|>\n")
                parts.append(f"<|im_start|>assistant\nVI: {vi}<|im_end|>\n")
                used_tokens += part_cost

        parts.append(f"<|im_start|>user\nJA: {text}<|im_end|>\n")
        parts.append("<|im_start|>assistant\nVI: ")
        return "".join(parts)

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        """Deprecated compatibility helper retained for tests and callers."""
        messages = [{"role": "system", "content": self.system_prompt}]
        few_shots = getattr(self, "_FEW_SHOT_EXAMPLES", ())

        max_output_tokens = min(self.max_tokens, max(50, len(text) * 3))
        token_budget = self.n_ctx - max_output_tokens
        used_tokens = len(self.system_prompt) // 3 + 10

        for user_ex, asst_ex in few_shots:
            cost = (len(user_ex) + len(asst_ex)) // 3 + 12
            if used_tokens + cost > token_budget - 80:
                break
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
            used_tokens += cost

        prompt_context_sentences = min(
            getattr(self, "context_sentences", 0),
            getattr(self, "_MAX_PROMPT_CONTEXT_SENTENCES", 1),
        )
        if self._keep_context and self._history and len(text) > 4 and prompt_context_sentences > 0:
            recent = list(self._history)[-prompt_context_sentences:]
            for jp, vi in recent:
                part_cost = (len(jp) + len(vi)) // 3 + 12
                if used_tokens + part_cost > token_budget - 40:
                    break
                messages.append({"role": "user", "content": f"JA: {jp}"})
                messages.append({"role": "assistant", "content": f"VI: {vi}"})
                used_tokens += part_cost

        messages.append({"role": "user", "content": f"JA: {text}"})
        messages.append({"role": "assistant", "content": "VI:"})
        return messages

    @staticmethod
    def _extract_translation(response: Any) -> str:
        if isinstance(response, dict):
            choices = response.get("choices") or []
        else:
            choices = getattr(response, "choices", []) or []
        if not choices:
            return ""

        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                content = choice.get("text")
        else:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None:
                content = getattr(choice, "text", None)

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    piece = item.get("text") or item.get("content") or ""
                else:
                    piece = getattr(item, "text", None) or getattr(item, "content", "")
                if piece:
                    parts.append(str(piece))
            return "".join(parts)
        return str(content or "")

    # Patterns that indicate LLM genuinely refused to translate (hard refusals)
    _HARD_REFUSAL_PATTERNS = (
        "tôi sẽ không dịch",
        "tôi không thể dịch",
        "tôi xin lỗi",
        "nội dung nhạy cảm",
        "không phù hợp",
        "i cannot translate",
        "i won't translate",
        "i can't translate",
    )

    # Patterns that indicate LLM added a preamble/explanation wrapper.
    # These are stripped from the output; if nothing useful remains, retry kicks in.
    _PREAMBLE_PATTERNS = (
        "tôi sẽ dịch",
        "tôi hiểu rồi",
        "bạn muốn tôi dịch",
        "bạn muốn tôi",
        "đoạn văn này",
        "đoạn hội thoại",
        "câu tiếng nhật",
        "bản dịch là",
        "hướng dẫn cho việc dịch",
        "đây là bản dịch",
        "bản dịch sang tiếng việt",
        "dịch sang tiếng việt:",
        "dịch đoạn",
    )

    _META_RESPONSE_PATTERNS = (
        "tôi là trợ lý",
        "tôi là một ai",
        "tôi là ai",
        "tôi có thể giúp",
        "bạn cần gì",
        "bạn có thể nói",
        "bạn có câu hỏi",
        "tôi sẽ giúp",
        "hãy cho tôi biết",
        "what can i help",
        "how can i assist",
        "i'm an ai",
        "as an ai",
        "i am an ai",
        "let me help",
        "here is the translation",
        "here's the translation",
    )

    # Combined for backward compatibility in detection
    _REFUSAL_PATTERNS = (
        _HARD_REFUSAL_PATTERNS
        + _PREAMBLE_PATTERNS
        + _META_RESPONSE_PATTERNS
        + (
            "có nghĩa là",
        )
    )

    # Regex to detect Japanese-specific characters (Hiragana, Katakana only)
    _JP_KANA_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
    # CJK Unified Ideographs (Kanji - shared with Chinese/Vietnamese)
    _CJK_RE = re.compile(r'[\u4E00-\u9FFF]')
    # Thai script (U+0E00-U+0E7F)
    _THAI_RE = re.compile(r'[\u0E00-\u0E7F]')
    # Korean Hangul (syllables + jamo + compatibility jamo)
    _HANGUL_RE = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]')
    # Arabic script
    _ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
    # Devanagari / Hindi
    _DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

    # Connective/incomplete sentence endings — these indicate the sentence
    # was cut off mid-thought and needs special handling in the prompt.
    _FRAGMENT_ENDINGS = (
        "だったりで", "たりで", "たりして",
        "ために", "ためで", "ため",
        "ので", "のに", "から",
        "けど", "けれど", "けれども",
        "ながら", "つつ", "たり",
        "して", "しつつ", "しながら",
        "として", "に対して", "について",
        "によって", "に関して",
        "で", "て", "に",
    )

    @staticmethod
    def _is_incomplete_fragment(text: str) -> bool:
        """Return True if input ends with a connective particle (incomplete sentence)."""
        text = text.strip()
        if not text:
            return False
        for ending in LlmTranslator._FRAGMENT_ENDINGS:
            if text.endswith(ending):
                # Avoid false positive: single-char endings need at least 4 chars
                if len(ending) <= 1 and len(text) < 4:
                    continue
                return True
        return False

    @staticmethod
    def _is_likely_english(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or LlmTranslator._VI_DIACRITICS_RE.search(cleaned):
            return False
        if LlmTranslator._JP_KANA_RE.search(cleaned) or LlmTranslator._CJK_RE.search(cleaned):
            return False

        words = cleaned.split()
        if len(words) < 2:
            return False

        ascii_words = LlmTranslator._ASCII_WORD_RE.findall(cleaned)
        if len(ascii_words) < 2:
            return False

        alpha_chars = [char for char in cleaned if char.isalpha()]
        if not alpha_chars:
            return False

        ascii_alpha_ratio = sum(
            1 for char in alpha_chars if char.isascii()
        ) / len(alpha_chars)
        ascii_word_ratio = len(ascii_words) / len(words)
        if ascii_alpha_ratio < 0.85 or ascii_word_ratio < 0.8:
            return False

        english_tokens: list[str] = []
        for word in ascii_words:
            english_tokens.extend(token.lower() for token in re.findall(r"[A-Za-z]+", word))
        english_hits = sum(
            1
            for token in english_tokens
            if token in LlmTranslator._COMMON_ENGLISH_WORDS
        )
        return english_hits >= 2 or len(ascii_words) >= 4

    @staticmethod
    def _strip_leading_english_word(text: str) -> str:
        """Strip leading English words/phrases from otherwise-Vietnamese output."""
        # Try stripping multiple leading English words (up to 8)
        words = text.split()
        if len(words) >= 2:
            # Find where Vietnamese starts (first word with Vietnamese diacritics or non-ASCII)
            vi_start = 0
            for i, word in enumerate(words[:8]):  # check first 8 words max
                stripped_word = word.rstrip('.,;:!?')
                # If word is purely ASCII and looks English, continue
                if stripped_word.isascii() and stripped_word.isalpha():
                    vi_start = i + 1
                else:
                    break

            if vi_start:
                leading_words = [word.rstrip('.,;:!?') for word in words[:vi_start]]
                should_strip = vi_start >= 2
                if vi_start == 1 and leading_words:
                    lead = leading_words[0]
                    should_strip = lead.lower() in LlmTranslator._LEADING_ENGLISH or (
                        lead.islower() and len(lead) > 4
                    )
                if should_strip:
                    rest = ' '.join(words[vi_start:]).lstrip(' ,').strip()
                    if rest and LlmTranslator._VI_DIACRITICS_RE.search(rest):
                        return rest

        match = re.match(r"^([A-Za-z]+)(?:,\s*|\s+)(.+)$", text)
        if not match:
            return text

        leading, rest = match.groups()
        if leading.lower() not in LlmTranslator._LEADING_ENGLISH:
            return text

        rest = rest.lstrip(" ,").strip()
        if not rest or not LlmTranslator._VI_DIACRITICS_RE.search(rest):
            return text
        return rest

    @staticmethod
    def _normalize_sentence_start(text: str, *, capitalize: bool) -> str:
        if not text:
            return text
        head = text[0].upper() if capitalize else text[0].lower()
        return head + text[1:]

    @classmethod
    def _fix_uoc_bias(cls, source: str, translation: str) -> str:
        cleaned_source = source.strip() if source else ""
        cleaned_translation = translation.strip() if translation else ""
        if not cleaned_translation or len(cleaned_source) <= 10:
            return cleaned_translation
        if any(marker in cleaned_source for marker in cls._LITERAL_UOC_SOURCE_HINTS):
            return cleaned_translation

        for prefix in cls._UOC_BIAS_PREFIXES:
            if not cleaned_translation.startswith(prefix):
                continue
            rest = cleaned_translation[len(prefix):].strip()
            if not rest:
                return cleaned_translation
            if "希望" in cleaned_source:
                fixed = f"Hi vọng {cls._normalize_sentence_start(rest, capitalize=False)}"
            else:
                fixed = cls._normalize_sentence_start(rest, capitalize=True)
            logger.info("Adjusted leading 'Ước' bias for %r -> %r", cleaned_source[:40], fixed[:80])
            return fixed
        return cleaned_translation

    @staticmethod
    def _clean_translation(text: str) -> str:
        cleaned = text.strip()
        prefixes = ("VI:", "Tiếng Việt:", "Bản dịch:", "Vietnamese:", "Dịch sang tiếng Việt:")
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        # Try extracting content after a colon (common preamble pattern)
        colon_index = cleaned.rfind(":")
        if colon_index != -1:
            meta_prefix = cleaned[:colon_index].strip()
            candidate = cleaned[colon_index + 1:].strip()
            if (
                meta_prefix
                and candidate
                and len(meta_prefix) < 50
                and any(word in meta_prefix.lower() for word in ("dịch", "bản", "sau", "tiếng"))
            ):
                cleaned = candidate
                for prefix in prefixes:
                    if cleaned.lower().startswith(prefix.lower()):
                        cleaned = cleaned[len(prefix):].strip()
            elif (
                meta_prefix
                and not candidate
                and any(word in meta_prefix.lower() for word in ("dịch", "bản", "sau", "tiếng"))
            ):
                # Preamble ending with colon but no content (cut off by stop token)
                logger.debug("Preamble with no content after colon: %r", cleaned)
                return ""
        # Handle multi-line: take first non-empty line that looks like a translation
        if "\n" in cleaned:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            # Skip lines that look like preambles
            for line in lines:
                lower_line = line.lower()
                is_preamble = any(
                    p in lower_line for p in LlmTranslator._PREAMBLE_PATTERNS
                ) or any(
                    p in lower_line for p in LlmTranslator._HARD_REFUSAL_PATTERNS
                ) or any(
                    p in lower_line for p in LlmTranslator._META_RESPONSE_PATTERNS
                )
                if not is_preamble:
                    cleaned = line
                    break
            else:
                # All lines are preambles/refusals
                return ""
        cleaned = cleaned.strip("`'\" ")
        # Reject if output contains wrong-language scripts
        if LlmTranslator._THAI_RE.search(cleaned):
            logger.warning("LLM output contains Thai script, rejecting: %r", cleaned[:80])
            return ""
        if LlmTranslator._HANGUL_RE.search(cleaned):
            logger.warning("LLM output contains Korean/Hangul, rejecting: %r", cleaned[:80])
            return ""
        if LlmTranslator._ARABIC_RE.search(cleaned):
            logger.warning("LLM output contains Arabic script, rejecting: %r", cleaned[:80])
            return ""
        if LlmTranslator._DEVANAGARI_RE.search(cleaned):
            logger.warning("LLM output contains Devanagari script, rejecting: %r", cleaned[:80])
            return ""
        # Reject if output contains Hiragana/Katakana (definitely JP, not translation)
        if LlmTranslator._JP_KANA_RE.search(cleaned):
            logger.warning("LLM output contains Japanese kana, rejecting: %r", cleaned[:80])
            return ""
        # If there are a few stray Kanji, strip them out instead of rejecting
        # (model sometimes leaves 1-2 kanji in an otherwise valid translation)
        if LlmTranslator._CJK_RE.search(cleaned):
            # Convert CJK numerals to Arabic digits before stripping
            cleaned = cleaned.translate(LlmTranslator._CJK_NUMERAL_MAP)
            stripped = LlmTranslator._CJK_RE.sub("", cleaned).strip()
            # Collapse multiple spaces left by removal
            stripped = re.sub(r" {2,}", " ", stripped)
            # If removing kanji leaves >60% of original length, use the stripped version
            if len(stripped) > len(cleaned) * 0.5 and len(stripped) > 5:
                logger.debug("Stripped stray kanji from output: %r -> %r", cleaned[:60], stripped[:60])
                cleaned = stripped
            else:
                # Too much kanji = model echoed input
                logger.warning("LLM output is mostly kanji, rejecting: %r", cleaned[:80])
                return ""
        # Reject refusals/explanations
        lower = cleaned.lower()
        for pattern in LlmTranslator._REFUSAL_PATTERNS:
            if pattern in lower:
                logger.warning("LLM refused/explained instead of translating: %r", cleaned)
                return ""
        # Reject if too long relative to reasonable translation (likely explanation)
        if len(cleaned) > 500:
            logger.warning("LLM output too long, likely explanation: %r", cleaned[:100])
            return ""
        # Reject repetitive gibberish (e.g. "yayayayaya...")
        if len(cleaned) > 20:
            words = cleaned.split()
            if words and len(set(words)) <= 2:
                logger.warning("LLM output is repetitive gibberish: %r", cleaned[:50])
                return ""
        if LlmTranslator._is_likely_english(cleaned):
            logger.warning("LLM output appears to be English, not Vietnamese: %r", cleaned[:80])
            return ""
        cleaned = LlmTranslator._strip_leading_english_word(cleaned)
        return cleaned


if __name__ == "__main__":
    translator = LlmTranslator()
    samples = [
        "本日のアジェンダを確認してください。",
        "デプロイのスケジュールについて話しましょう。",
        "バックログのリファインメントは来週です。",
    ]
    for s in samples:
        print(f"JP: {s}")
        print(f"VI: {translator.translate(s)}")
        print()
