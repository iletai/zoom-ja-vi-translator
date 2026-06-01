"""LLM-based Japanese-to-Vietnamese translation via llama-cpp-python."""
from __future__ import annotations

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
    "Bạn là phiên dịch viên chuyên nghiệp Nhật-Việt trong meeting IT. "
    "QUY TẮC: "
    "1. Chỉ xuất bản dịch tiếng Việt, không giải thích. "
    "2. Giữ nguyên thuật ngữ tiếng Anh: Cloud, AWS, API, deploy, sprint, IoT, AI, Lambda, EC2, S3. "
    "3. Dịch ngắn gọn, tự nhiên. Bỏ kính ngữ thừa. "
    "4. Tên riêng giữ nguyên romaji: Akihabara, Shibuya, Tokyo."
)


class LlmTranslator:
    """Drop-in Japanese-to-Vietnamese translator backed by llama.cpp."""

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
        self.n_batch = max(1, int(getattr(config, "LLM_N_BATCH", 512)))
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

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            use_mlock=bool(getattr(config, "LLM_USE_MLOCK", False)),
            verbose=False,
        )

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
        "お疲れ様です": "Xin chào",
        "お疲れ様でした": "Cảm ơn đã vất vả",
        "よろしくお願いします": "Xin vui lòng hỗ trợ",
        "よろしくお願いいたします": "Rất mong được hỗ trợ",
        "ありがとうございます": "Cảm ơn",
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

    def _translate_one(self, text: str, update_context: bool = True) -> str:
        cleaned = text.strip() if text else ""
        if not cleaned:
            return ""

        # Check filler words first (no LLM needed)
        filler_result = self._FILLER_MAP.get(cleaned)
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
                messages = self._build_messages(processed)
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=0.1,
                    max_tokens=self.max_tokens,
                    stop=["\n", "\n\n", "<|im_end|>"],
                )
                translation = self._clean_translation(self._extract_translation(response))
                if not translation:
                    retry_messages = [
                        {
                            "role": "system",
                            "content": "Dịch tiếng Nhật sang tiếng Việt. Chỉ xuất bản dịch.",
                        },
                        {"role": "user", "content": processed},
                    ]
                    retry_response = self.llm.create_chat_completion(
                        messages=retry_messages,
                        temperature=0.3,
                        top_p=0.5,
                        frequency_penalty=0.2,
                        max_tokens=self.max_tokens,
                        stop=["\n", "\n\n", "<|im_end|>"],
                    )
                    translation = self._clean_translation(
                        self._extract_translation(retry_response)
                    )
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

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        # Fixed one-shot example to anchor Vietnamese output format.
        messages.append(
            {
                "role": "user",
                "content": "Dịch sang tiếng Việt: クラウドのarchitectureについて説明します",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Tôi sẽ giải thích về kiến trúc Cloud.",
            }
        )
        # Skip context for very short inputs — small models tend to repeat
        # previous translations when the current input is just 1-3 chars.
        context_line = ""
        if self._keep_context and self._history and len(text) > 4:
            recent = list(self._history)[-self.context_sentences:]
            context_line = "".join(
                f"(前: {jp} → {vi})\n" for jp, vi in recent
            )
        # Current sentence to translate — Vietnamese command biases output language
        user_content = f"{context_line}Dịch sang tiếng Việt: {text}"
        messages.append({"role": "user", "content": user_content})
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

    # Patterns that indicate LLM refused or explained instead of translating
    _REFUSAL_PATTERNS = (
        "tôi sẽ không dịch",
        "tôi không thể dịch",
        "tôi sẽ dịch",
        "đoạn văn này",
        "xin chào",
        "chào bạn",
        "có nghĩa là",
        "câu tiếng nhật",
        "bản dịch là",
        "hướng dẫn cho việc dịch",
        "đây là bản dịch",
        "bản dịch sang tiếng việt",
        "dịch sang tiếng việt:",
        "i cannot translate",
        "i won't translate",
        "i can't translate",
    )

    # Regex to detect Japanese-specific characters (Hiragana, Katakana only)
    _JP_KANA_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
    # CJK Unified Ideographs (Kanji - shared with Chinese/Vietnamese)
    _CJK_RE = re.compile(r'[\u4E00-\u9FFF]')

    @staticmethod
    def _clean_translation(text: str) -> str:
        cleaned = text.strip()
        prefixes = ("VI:", "Tiếng Việt:", "Bản dịch:", "Vietnamese:", "Dịch sang tiếng Việt:")
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
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
        # Take first line only
        if "\n" in cleaned:
            cleaned = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
        cleaned = cleaned.strip("`'\" ")
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
        # Reject full English output (no Vietnamese diacritics + many words)
        words = cleaned.split()
        if len(words) > 5 and not LlmTranslator._VI_DIACRITICS_RE.search(cleaned):
            logger.warning("LLM output appears to be English, not Vietnamese: %r", cleaned[:80])
            return ""
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
