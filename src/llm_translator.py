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
from src.japanese_names import SURNAME_MAP, SURNAME_SET, KATAKANA_NAMES
from src.sentence_aggregator import split_japanese_sentences

# Multi-character surnames: safe for substring matching.
# Single-character surnames (森, 林, 関) excluded to avoid false positives
# with common kanji (関係, 関数, 森林). These are matched only via さん suffix.
_LONG_SURNAMES = {s for s in SURNAME_SET if len(s) >= 2}

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
    # English first — Qwen2.5's strongest instruction-following pathway
    "You are a Japanese-to-Vietnamese translator for IT meetings "
    "about emergency medical dispatch systems (救急搬送システム) for Japan's fire/EMS service. "
    "CRITICAL: Output ONLY Vietnamese using Latin script. "
    "NEVER use Chinese characters (漢字/汉字). NEVER use Japanese kana. "
    "Output exactly ONE line of Vietnamese translation.\n"
    # Vietnamese reinforcement
    "Bạn là máy dịch Nhật→Việt cho cuộc họp CNTT về hệ thống điều phối cứu hộ Nhật Bản. "
    "CHỈ xuất tiếng Việt (chữ Latin). "
    "KHÔNG ĐƯỢC dùng chữ Hán/tiếng Trung/tiếng Nhật. "
    "Giữ nguyên thuật ngữ IT: Cloud, AWS, API, deploy, sprint, Lambda, EC2, S3. "
    "Dịch thuật ngữ cứu hộ: 消防＝cứu hỏa, 救急＝cấp cứu, 搬送＝vận chuyển, "
    "傷病者＝nạn nhân, 引き継ぎ＝bàn giao, 出動＝xuất kích/điều động, "
    "受入＝tiếp nhận. "
    "Dịch ngắn gọn, tự nhiên.\n"
    "QUAN TRỌNG - Giữ tên người Nhật (romaji):\n"
    "  - Tên + さん/様 KHÔNG ĐƯỢC bỏ hoặc đổi thành 'bạn'/'ông'/'cô'\n"
    "  - '中野さん' → 'anh Nakano' / 'Nakano-san'\n"
    "  - '川村さんのほう' → 'phía Kawamura'\n"
    "  - '羽根さん' → 'anh Hane'\n"
    "  - 'カリスさん' → 'anh Caris'\n"
    "  - 'ハレ井さん' → 'anh Harei'\n"
    "  - 'ジャンさん' → 'anh Jan'\n"
    "  - 'ハレ井さん' → 'anh Harei'\n"
    "  - Tên Nhật Kanji (深瀬, 大森, 河合) giữ romaji: Fukase, Omori, Kawai\n"
    "CẢNH BÁO - Không tự ý tách kanji ghép:\n"
    "  - 関係 = quan hệ (KHÔNG tách thành 関 Seki)\n"
    "  - 関数 = hàm số (KHÔNG tách thành 関 Seki)\n"
    "  - 関連 = liên quan (KHÔNG tách thành 関 Seki)\n"
    "  - Đây là từ ghép thông thường, KHÔNG phải tên riêng\n"
    "CẢNH BÁO - Dịch sát, không bịa:\n"
    "  - Nếu câu nhập vô nghĩa (ASR lỗi), dịch sát nghĩa đen\n"
    "  - KHÔNG ĐƯỢC thêm ngữ cảnh không có trong câu gốc\n"
    "  - Ưu tiên dịch sát > dịch hay"
)


class LlmTranslator:
    """Drop-in Japanese-to-Vietnamese translator backed by llama.cpp."""

    _FEW_SHOT_EXAMPLES = (
        ("JA: 次のsprintでAPIを修正します", "VI: Chúng tôi sẽ sửa API trong sprint tới."),
        ("JA: この処理を入れることによってエラーが起きない対応を入れました",
         "VI: Bằng cách thêm xử lý này, chúng tôi đã thêm biện pháp để không xảy ra lỗi."),
        ("JA: 木曜日の午後にミーティングがあります",
         "VI: Có cuộc họp vào chiều thứ Năm."),
        ("JA: 後でスケジュールを確認します",
         "VI: Tôi sẽ kiểm tra lịch trình sau."),
        ("JA: 来月のスケジュールを共有します",
         "VI: Tôi sẽ chia sẻ lịch trình tháng tới."),
        ("JA: 確認作業が完了しました", "VI: Công việc xác nhận đã hoàn thành."),
    )

    # Day-of-week pre-processing dictionary (deterministic, zero-latency)
    _JP_DOW_MAP = {
        # Full forms (〇曜日)
        "月曜日": "thứ Hai", "火曜日": "thứ Ba", "水曜日": "thứ Tư",
        "木曜日": "thứ Năm", "金曜日": "thứ Sáu",
        "土曜日": "thứ Bảy", "日曜日": "Chủ nhật",
        # Short forms (〇曜) — ASR often drops 日
        "月曜": "thứ Hai", "火曜": "thứ Ba", "水曜": "thứ Tư",
        "木曜": "thứ Năm", "金曜": "thứ Sáu",
        "土曜": "thứ Bảy", "日曜": "Chủ nhật",
    }

    # SVO word order fix patterns — Vietnamese: Subject + Modal + Verb
    _SVO_FIX_PATTERNS = [
        # "sẽ tôi" → "tôi sẽ" etc.
        (re.compile(r'\bsẽ\s+(tôi|chúng tôi|chúng ta|bạn|họ|anh|chị)\b', re.IGNORECASE),
         lambda m: f"{m.group(1)} sẽ"),
        (re.compile(r'\bđã\s+(tôi|chúng tôi|chúng ta|bạn|họ|anh|chị)\b', re.IGNORECASE),
         lambda m: f"{m.group(1)} đã"),
        (re.compile(r'\bđang\s+(tôi|chúng tôi|chúng ta|bạn|họ|anh|chị)\b', re.IGNORECASE),
         lambda m: f"{m.group(1)} đang"),
    ]

    # Wrong Ư-starter words (almost never valid at sentence start in IT context)
    _WRONG_U_STARTERS = ("Ướt ", "Ưỡn ", "Ướm ")
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
        self.n_ctx = max(128, int(getattr(config, "LLM_N_CTX", 512)))
        self.n_threads = max(
            1,
            int(getattr(config, "LLM_N_THREADS", getattr(config, "_PHYSICAL_CORES", 1) or 1)),
        )
        self.n_batch = max(1, int(getattr(config, "LLM_N_BATCH", 512)))
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

        # Build llama kwargs with performance optimizations
        import multiprocessing
        _logical = multiprocessing.cpu_count()

        llama_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_threads_batch": _logical,  # Prefill: use all logical cores
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
        # KV cache quantization (Q8_0 = type 8): ~50% VRAM savings, negligible quality loss
        if "type_k" in llama_params:
            llama_kwargs["type_k"] = 8
            llama_kwargs["type_v"] = 8

        # Speculative decoding via prompt-lookup (zero extra model cost)
        try:
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
            if "draft_model" in llama_params:
                num_pred = 10 if self.n_gpu_layers != 0 else 2
                llama_kwargs["draft_model"] = LlamaPromptLookupDecoding(
                    num_pred_tokens=num_pred,
                    max_ngram_size=2,
                )
                logger.info("Speculative decoding enabled (num_pred_tokens=%d)", num_pred)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Speculative decoding not available: %s", exc)

        self.llm = Llama(**llama_kwargs)

        # Enable RAM cache for KV state reuse across calls
        try:
            from llama_cpp import LlamaRAMCache
            cache_mb = max(64, config.LLM_RAM_CACHE_MB)
            self.llm.set_cache(LlamaRAMCache(capacity_bytes=cache_mb * 1024 * 1024))
            logger.info("LlamaRAMCache enabled (%dMB)", cache_mb)
        except (ImportError, AttributeError, Exception) as exc:
            logger.debug("LlamaRAMCache not available: %s", exc)

        # Build logit bias to penalize Chinese-specific tokens.
        self._chinese_logit_bias = self._build_chinese_logit_bias()

        # Optionally build GBNF grammar for hard Latin-only output constraint.
        self._vi_grammar = self._build_vi_grammar()

        # Merge common Japanese surnames into proper noun map (romaji copy-through)
        self._PROPER_NOUN_MAP.update(SURNAME_MAP)

        # NLLB fast-path translator for simple sentences
        self._fast_translator = self._init_fast_translator()

        try:
            self.warmup()
        except Exception as exc:  # pragma: no cover - best-effort latency optimization
            logger.warning("LLM translator warmup failed: %s", exc)

    def _build_chinese_logit_bias(self) -> dict[int, float]:
        """Build token-level logit bias hard-blocking CJK and Japanese kana tokens.

        Iterates the entire model vocabulary and identifies tokens whose decoded
        text is composed entirely of:
        - CJK Unified Ideographs (Chinese characters / kanji)
        - Hiragana or Katakana (Japanese-specific)

        These get -100.0 logit bias (effectively impossible to sample).
        This targets the root cause: prevents both Chinese AND Japanese echo
        from being generated. Returns empty dict on failure (graceful degradation).
        """
        bias: dict[int, float] = {}
        try:
            n_vocab_fn = getattr(self.llm, "n_vocab", None)
            detokenize_fn = getattr(self.llm, "detokenize", None)
            if not n_vocab_fn or not detokenize_fn:
                return {}
            vocab_size = n_vocab_fn()
            for token_id in range(vocab_size):
                try:
                    raw = detokenize_fn([token_id])
                    if not raw:
                        continue
                    text = raw.decode("utf-8", errors="ignore")
                    if not text:
                        continue
                    # Only consider non-whitespace characters
                    chars = [c for c in text if not c.isspace()]
                    if not chars:
                        continue
                    # Block if ALL characters are CJK/Hiragana/Katakana
                    if all(
                        "\u4E00" <= c <= "\u9FFF"       # CJK Unified Ideographs
                        or "\u3400" <= c <= "\u4DBF"    # CJK Extension A
                        or "\uF900" <= c <= "\uFAFF"    # CJK Compatibility
                        or "\u3040" <= c <= "\u309F"    # Hiragana
                        or "\u30A0" <= c <= "\u30FF"    # Katakana
                        or "\u3000" <= c <= "\u303F"    # CJK Symbols/Punctuation
                        for c in chars
                    ):
                        bias[token_id] = -100.0
                except Exception:
                    pass
            logger.info("CJK/kana logit bias: hard-blocking %d tokens from vocab of %d", len(bias), vocab_size)
        except Exception as exc:
            logger.warning("Could not build CJK logit bias: %s", exc)
        return bias

    def _build_vi_grammar(self):
        """Build GBNF grammar constraining output to Latin/Vietnamese chars only.

        Only active when config.LLM_USE_GRAMMAR is True. Provides a hard guarantee
        against CJK output at the cost of ~10-30ms extra latency per token.
        """
        if not getattr(config, "LLM_USE_GRAMMAR", False):
            return None
        try:
            from llama_cpp import LlamaGrammar
        except ImportError:
            logger.debug("LlamaGrammar not available, skipping grammar constraint")
            return None
        # GBNF grammar: allow Latin, Vietnamese diacritics, digits, punctuation, spaces
        gbnf = r'''
root ::= token+
token ::= vichar | space | punct | digit
vichar ::= [a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]
space ::= [ \t]
punct ::= [.,;:!?\-'"()\[\]{}/\\@#$%&*+=_~<>|]
digit ::= [0-9]
'''
        try:
            grammar = LlamaGrammar.from_string(gbnf.strip())
            logger.info("GBNF Vietnamese grammar constraint enabled")
            return grammar
        except Exception as exc:
            logger.warning("Failed to build GBNF grammar: %s", exc)
            return None

    def _init_fast_translator(self):
        """Initialize NLLB fast-path translator for simple sentences."""
        try:
            from src.translator import NllbTranslator
            fast = NllbTranslator()
            logger.info("NLLB fast-path translator initialized")
            return fast
        except Exception as exc:
            logger.debug("NLLB fast-path not available: %s", exc)
            return None

    # Keigo/complex grammar indicators that require LLM
    _COMPLEX_GRAMMAR_MARKERS = (
        "については", "に関して", "に対して",
        "というのは", "ということで", "ということ",
        "ではないでしょうか", "させていただ",
        "いただけ", "くださ", "ございま",
    )

    def _classify_complexity(self, text: str) -> str:
        """Route sentence to appropriate translation tier.

        Returns: 'nllb' for fast NMT path, 'llm' for full LLM translation.
        """
        if not getattr(self, "_fast_translator", None):
            return "llm"
        n = len(text)
        # Keigo/formal patterns → need LLM for nuance (check first, any length)
        if any(marker in text for marker in self._COMPLEX_GRAMMAR_MARKERS):
            return "llm"
        # Person name detected → LLM (NLLB hallucinates literal kanji meanings)
        if "さん" in text or "様" in text:
            return "llm"
        # Multi-char surnames substring match (single-char: 森/林/関 excluded
        # to avoid false positives with common kanji compounds like 関係/関数)
        if any(name in text for name in _LONG_SURNAMES):
            return "llm"
        # Very short fragments (fillers, simple nouns) → NLLB is fine
        if n <= 30:
            return "nllb"
        # Longer text or anything ambiguous → LLM for context-aware quality
        return "llm"

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
        "うんうんうん": "Vâng vâng vâng",
        "はい": "Vâng",
        "はいはい": "Vâng, vâng",
        "はいはいはい": "Vâng vâng vâng",
        "はいもね": "Vâng, đúng nhỉ",
        "ええ": "Vâng",
        "え": "Ơ",
        "えっと": "À...",
        "あの": "À...",
        "あのう": "À...",
        "ああ": "À",
        "あ": "À",
        "まあ": "Thôi thì",
        "なるほど": "Ra vậy",
        "なるほどね": "Ra vậy nhỉ",
        "そうですね": "Đúng vậy nhỉ",
        "そうそう": "Đúng, đúng",
        "そうそうそう": "Đúng đúng đúng",
        "ですね": "Đúng vậy",
        "っていう": "Nghĩa là",
        "ます": "...",
        "ねえ": "Này",
        "ねえねえ": "Này này",
        "うんねえねえ": "Vâng, này này",
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
        # Emergency medical / rescue domain katakana terms
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
        # Emergency medical / rescue domain
        "搬送": "vận chuyển",
        "搬送元": "nơi xuất phát vận chuyển",
        "搬送決定": "quyết định vận chuyển",
        "傷病者": "nạn nhân",
        "引き継ぎ": "bàn giao",
        "出動": "xuất kích/điều động",
        "指令": "chỉ thị/điều phối",
        "救急搬送": "vận chuyển cấp cứu",
        "現場": "hiện trường",
        "災害": "thảm họa",
        "患者": "bệnh nhân",
        "受入": "tiếp nhận",
        "消防": "cứu hỏa",
        "救助": "cứu hộ",
        "案件": "hạng mục",
        "着手": "bắt đầu triển khai",
        "進捗": "tiến độ",
        "受け入れ": "tiếp nhận",
        "負荷試験": "kiểm tra tải",
        "レビュー": "review",
        "設計": "thiết kế",
        "Ｃross-Tenant": "Cross-Tenant",
        "クロステナント": "Cross-Tenant",
        "マルチテナント": "multi-tenant",
        "ステータス": "trạng thái",
        "ダッシュボード": "dashboard",
        "リリース": "release",
        "トリアージ": "triage",
        "バイタル": "vital signs",
        "インシデント": "incident",
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
        # Emergency/rescue organizations (Japan)
        "東京消防庁": "Sở Cứu hỏa Tokyo",
        "消防庁": "Cục Cứu hỏa",
        "消防本部": "Sở chỉ huy Cứu hỏa",
        "消防団": "Đội Cứu hỏa",
        "消防署": "Trạm Cứu hỏa",
        "消防": "Cứu hỏa",
        "救急": "Cấp cứu",
        "救急隊": "Đội Cấp cứu",
        "救急車": "Xe Cấp cứu",
        "警察": "Cảnh sát",
        "自衛隊": "Lực lượng Phòng vệ",
        "日本赤十字社": "Hội Chữ thập đỏ Nhật Bản",
        "DMAT": "DMAT",
        "医療機関": "Cơ sở y tế",
        "病院": "Bệnh viện",
        # Disaster/incident management systems
        "EMIS": "EMIS",
        "広域災害": "Thảm họa diện rộng",
        "広域地図": "bản đồ diện rộng",
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

        # Pre-process: replace 姓+さん with romaji-san to prevent name hallucination
        # Sort by kanji length (longest first) to handle overlapping surnames
        for kanji, romaji in sorted(SURNAME_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            # Only process multi-char surnames for substring replacement
            # (single-char: 森/林/関 excluded to avoid false positives in 関係/関数)
            if len(kanji) < 2:
                continue
            suffix_with = f"{kanji}さん"
            if suffix_with in cleaned:
                cleaned = cleaned.replace(suffix_with, f"{romaji}-san")
            suffix_sama = f"{kanji}様"
            if suffix_sama in cleaned:
                cleaned = cleaned.replace(suffix_sama, f"{romaji}-sama")
            # Replace bare kanji surname with romaji (safe for multi-char names)
            if kanji in cleaned:
                cleaned = cleaned.replace(kanji, romaji)
        # Also pre-process katakana names (guest names from meeting evidence)
        for kana, romaji in sorted(KATAKANA_NAMES.items(), key=lambda x: len(x[0]), reverse=True):
            if kana in cleaned:
                cleaned = cleaned.replace(kana, romaji)

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

        # Pre-process: replace day-of-week kanji with correct Vietnamese (deterministic)
        for jp_day, vi_day in sorted(
            self._JP_DOW_MAP.items(), key=lambda x: -len(x[0])
        ):
            processed = processed.replace(jp_day, vi_day)

        # Pre-process: replace known katakana IT terms with equivalents. Longest first.
        for ja_term, en_term in sorted(
            self._KATAKANA_TERM_MAP.items(), key=lambda x: -len(x[0])
        ):
            processed = processed.replace(ja_term, en_term)

        # Pre-process common misrecognized patterns
        processed = processed.replace("百パー", "100%")
        processed = processed.replace("百%", "100%")

        # NLLB fast-path: route simple sentences to NMT for lower latency
        tier = self._classify_complexity(cleaned)
        if tier == "nllb":
            try:
                nllb_result = self._fast_translator.translate(processed)
                if nllb_result and nllb_result.strip():
                    result = self._fix_word_order(nllb_result.strip())
                    if update_context and self._keep_context:
                        self._history.append((cleaned, result))
                    logger.debug("NLLB fast-path: %r -> %r", cleaned[:40], result[:60])
                    return result
            except Exception as exc:
                logger.debug("NLLB fast-path failed, falling back to LLM: %s", exc)

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
                    logit_bias=self._chinese_logit_bias or None,
                    grammar=self._vi_grammar,
                    stop=["\n", "<|im_end|>", "JA:", "JP:", "\nJA:",
                          " Here ", " This is", "I am", "I will", "Let me",
                          # Chinese function words — stop generation immediately
                          # if model switches to Chinese
                          "的", "是", "了", "在", "不", "我们",
                          "这", "那", "有", "没", "就", "也", "都",
                          "会", "能", "要", "说", "对", "让", "把",
                          "被", "给", "从", "还", "很", "到", "过",
                          "吗", "吧", "呢", "啊", "嗯", "做", "看",
                          "为", "与", "或", "而", "且", "因", "所",
                          ],
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
                        logit_bias=self._chinese_logit_bias or None,
                        grammar=self._vi_grammar,
                        stop=["<|im_end|>", "\n\n", "JA:", "JP:",
                              " Here ", " This is", "I am", "I will", "Let me",
                              "的", "是", "了", "在", "不", "我们",
                              "这", "那", "有", "没", "就", "也", "都",
                              "会", "能", "要", "说", "对", "让", "把",
                              "被", "给", "从", "还", "很", "到", "过",
                              "吗", "吧", "呢", "啊", "嗯", "做", "看",
                              "为", "与", "或", "而", "且", "因", "所",
                              ],
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
        "ですけど", "ですが", "ですけれど",
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

        # Fix wrong-U starters (Ướt, Ưỡn, Ướm) that are hallucinated
        for wrong in cls._WRONG_U_STARTERS:
            if cleaned_translation.startswith(wrong):
                rest = cleaned_translation[len(wrong):].strip()
                if rest:
                    fixed = cls._normalize_sentence_start(rest, capitalize=True)
                    logger.info("Stripped wrong-U starter %r: %r -> %r", wrong.strip(), cleaned_translation[:40], fixed[:60])
                    return fixed
        return cleaned_translation

    @staticmethod
    def _fix_word_order(text: str) -> str:
        """Fix SVO word order errors where modal comes before subject."""
        result = text
        for pattern, replacement in LlmTranslator._SVO_FIX_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

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
            # Accept if stripped result has Vietnamese diacritics (proof it's real VI)
            has_vi_diacritics = bool(LlmTranslator._VI_DIACRITICS_RE.search(stripped))
            # If removing kanji leaves useful text, use the stripped version
            if has_vi_diacritics and len(stripped) >= 1:
                logger.debug("Stripped stray kanji (VI confirmed): %r -> %r", cleaned[:60], stripped[:60])
                cleaned = stripped
            elif len(stripped) > len(cleaned) * 0.4 and len(stripped) > 3:
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
        # Fix SVO word order issues before returning
        cleaned = LlmTranslator._fix_word_order(cleaned)
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
