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
    "Bạn là mô hình dịch thuật chuyên nghiệp trong lĩnh vực CNTT. "
    "Dịch chính xác từ tiếng Nhật sang tiếng Việt. "
    "Giữ nguyên thuật ngữ kỹ thuật: Cloud, API, deploy, sprint, AWS, IoT, AI, "
    "on-premises, architect, solution, enterprise, public sector. "
    "Chỉ xuất bản dịch, không giải thích thêm."
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
    }

    # Katakana IT terms → English. Used as pre-processing substitution (mid-sentence).
    # Sorted by length (longest first) at substitution time to avoid partial matches.
    _KATAKANA_TERM_MAP = {
        "ソリューションアーキテクト": "Solution Architect",
        "クラウドファースト": "Cloud First",
        "マイクロサービス": "microservice",
        "パブリックセクター": "public sector",
        "テクノロジー": "Technology",
        "ウェブサービス": "Web Services",
        "オンプレミス": "on-premises",
        "エンタープライズ": "enterprise",
        "マイグレーション": "migration",
        "サーバーレス": "serverless",
        "アーキテクト": "architect",
        "エンジニア": "engineer",
        "バックログ": "backlog",
        "スプリント": "sprint",
        "ロボット": "robot",
        "アマゾン": "Amazon",
        "クラウド": "Cloud",
        "オンプレ": "on-premises",
        "デプロイ": "deploy",
        "インフラ": "infrastructure",
        "コンテナ": "container",
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

        # Pre-process: replace known katakana IT terms with English equivalents
        # so the LLM doesn't mistranslate them. Longest match first.
        processed = cleaned
        for ja_term, en_term in sorted(
            self._KATAKANA_TERM_MAP.items(), key=lambda x: -len(x[0])
        ):
            processed = processed.replace(ja_term, en_term)

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
                    logger.warning("Empty LLM translation for JP sentence: %r", cleaned)
                    return ""
                # Only add valid translations to context (prevents pollution)
                if update_context and self._keep_context and len(translation) < 300:
                    self._history.append((cleaned, translation))
                return translation
        except Exception as exc:
            logger.warning("LLM translation failed for %r: %s", cleaned, exc)
            return ""

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
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
        for prefix in ("VI:", "Tiếng Việt:", "Bản dịch:", "Vietnamese:", "Dịch sang tiếng Việt:"):
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
