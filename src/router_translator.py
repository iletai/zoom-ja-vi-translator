"""JA→Vietnamese translation through a local OpenAI-compatible gateway (9router).

This is a drop-in alternative to ``NllbTranslator`` / ``LlmTranslator``: it
exposes the same ``translate`` / ``translate_many`` / ``warmup`` surface the
pipeline calls, but instead of running a model in-process it POSTs to a local
gateway (``http://127.0.0.1:20128/v1`` by default) that fronts hosted models
(Claude, GPT, DeepSeek, …). Activate with ``ZT_TRANSLATOR=router``.

Trade-off vs. the local backends: far higher translation quality and zero model
download / RAM, at the cost of per-segment network latency and *not* being
offline — audio text leaves the machine through the gateway. Use ``nllb``/``llm``
when fully-local operation is required.

Even with a strong hosted model, a thin layer of deterministic domain
pre-processing pays off for THIS meeting domain (IT + Japanese emergency
dispatch): person names are romanized so "深瀬" never becomes a literal "deep
rapids", katakana IT loanwords and proper nouns are substituted, and pure filler
("はい") skips the network entirely. These maps are the shared single source of
truth (``src/domain_data`` + ``src/japanese_names``), so they stay in sync with
the local LLM backend.

Robustness matters because this sits on the latency-sensitive translate path:
every call has a hard timeout and one retry; a batch is translated concurrently
so N sentences cost ~1 round-trip, not N; and a slow or flapping gateway degrades
to a dropped line rather than stalling the meeting.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

import config
from src.domain_data import KATAKANA_TERMS, PROPER_NOUNS
from src.japanese_names import KATAKANA_NAMES, SURNAME_MAP
from src.post_correction import post_correct
from src.sentence_aggregator import split_japanese_sentences

logger = logging.getLogger(__name__)


def _is_latin(s: str) -> bool:
    """True if ``s`` is plain ASCII (romaji / English / acronym, no diacritics).

    Used to decide what is safe to substitute *into* a still-Japanese source
    sentence: Latin replacements (Fukase-san, deploy, DMAT, Tokyo) copy through
    cleanly, but a Vietnamese replacement (傷病者→"Nạn nhân") dropped between
    Japanese particles makes a half-translated hybrid that confuses the model —
    those are left as kanji for the LLM to translate, and surfaced via the
    system-prompt glossary instead.
    """
    return s.isascii()


# Domain glossary injected into the system prompt (not substituted into the
# source). These are the kanji/terms whose Vietnamese rendering we want the model
# to use; leaving them as Japanese in the source and steering via the prompt
# avoids the hybrid-sentence problem while still pinning terminology. Sourced
# from the shared PROPER_NOUNS / DOMAIN_TERMS single source of truth.
def _build_glossary_block() -> str:
    from src.domain_data import DOMAIN_TERMS

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    # Prefer DOMAIN_TERMS (lowercase Vietnamese) then non-Latin PROPER_NOUNS.
    for jp, vi in list(DOMAIN_TERMS.items()) + list(PROPER_NOUNS.items()):
        if jp in seen or _is_latin(vi):
            continue
        seen.add(jp)
        pairs.append((jp, vi))
    lines = "; ".join(f"{jp}={vi}" for jp, vi in pairs)
    return lines


# Pre-sorted longest-first so multi-char terms win over their own substrings
# (e.g. "クロステナント" before "テナント"). Built once at import.
_SORTED_SURNAMES = sorted(SURNAME_MAP.items(), key=lambda kv: -len(kv[0]))
_SORTED_KATAKANA_NAMES = sorted(KATAKANA_NAMES.items(), key=lambda kv: -len(kv[0]))
# Only Latin-valued substitutions are safe to inject into a Japanese source.
_SORTED_KATAKANA_TERMS = sorted(
    ((k, v) for k, v in KATAKANA_TERMS.items() if _is_latin(v)), key=lambda kv: -len(kv[0])
)
_SORTED_PROPER_NOUNS = sorted(
    ((k, v) for k, v in PROPER_NOUNS.items() if _is_latin(v)), key=lambda kv: -len(kv[0])
)

# Pure back-channel / filler utterances: translating these through an LLM both
# wastes a round-trip and tends to over-expand them ("はい" → "Vâng được ạ").
# An exact-match lookup returns the canonical short form and skips the network.
_FILLER_MAP = {
    "うん": "Vâng", "うんうん": "Vâng, vâng", "はい": "Vâng", "はいはい": "Vâng, vâng",
    "ええ": "Vâng", "え": "Ơ", "えっと": "À...", "あの": "À...", "あのう": "À...",
    "ああ": "À", "あ": "À", "まあ": "Thôi thì", "なるほど": "Ra vậy",
    "なるほどね": "Ra vậy nhỉ", "そうですね": "Đúng vậy nhỉ", "そうそう": "Đúng, đúng",
    "ですね": "Đúng vậy", "こんにちは": "Xin chào", "おはようございます": "Chào buổi sáng",
}


def _looks_like_refusal_or_echo(src: str, out: str) -> bool:
    """Reject outputs that are clearly not a translation.

    Hosted chat models occasionally prepend boilerplate ("Here is the
    translation:") or echo the Japanese back. We only catch the unambiguous
    cases — a leaked instruction sentence, or output identical to the source —
    and let the normal text through untouched.
    """
    if not out:
        return True
    low = out.lower()
    if low.startswith(("here is", "translation:", "вот", "sure,", "câu dịch", "bản dịch:")):
        return True
    # Echoed the source verbatim (model declined to translate).
    return out.strip() == src.strip()


class RouterTranslator:
    """Translate Japanese to Vietnamese via the 9router OpenAI-compatible API."""

    def __init__(self) -> None:
        self.base_url = str(config.ROUTER_BASE_URL).rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = str(config.ROUTER_MODEL)
        self.api_key = str(config.ROUTER_API_KEY)
        # Pin domain terminology by appending the shared Vietnamese glossary to
        # the system prompt (terms left as kanji in the source — see _preprocess).
        base_prompt = str(config.ROUTER_SYSTEM_PROMPT)
        glossary = _build_glossary_block()
        self.system_prompt = (
            f"{base_prompt}\n\nThuật ngữ chuyên ngành (dùng đúng các bản dịch sau): {glossary}"
            if glossary else base_prompt
        )
        self.temperature = float(config.ROUTER_TEMPERATURE)
        self.max_tokens = int(config.ROUTER_MAX_TOKENS)
        self.timeout = float(config.ROUTER_TIMEOUT_S)
        self.context_sentences = max(0, int(config.ROUTER_CONTEXT_SENTENCES))
        self.max_parallel = max(1, int(getattr(config, "ROUTER_MAX_PARALLEL", 4)))

        self._keep_context = self.context_sentences > 0
        self._history: deque[tuple[str, str]] = deque(
            maxlen=self.context_sentences if self._keep_context else 1
        )
        self._lock = threading.Lock()

        # A pooled session keeps the TCP/TLS connection warm across segments,
        # shaving the per-call handshake off the latency path.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.info("RouterTranslator → %s (model=%s)", self.url, self.model)

        # Prime the TLS connection + model now so the first LIVE segment isn't
        # the one that eats the ~2s handshake. Best-effort, like NLLB/LLM.
        try:
            self.warmup()
        except Exception as exc:  # pragma: no cover - best-effort latency optimization
            logger.warning("RouterTranslator warmup failed: %s", exc)

    # ---- public surface (mirrors NllbTranslator / LlmTranslator) ----------- #

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        if not getattr(config, "TRANSLATE_SPLIT_SENTENCES", True):
            return self._translate_one(text)
        sentences = split_japanese_sentences(text)
        if len(sentences) <= 1:
            return self._translate_one(sentences[0] if sentences else text)
        # Multi-sentence: translate concurrently, then join in order. Context is
        # not threaded between them (they were spoken together) to keep the
        # fan-out race-free; see translate_many.
        parts = self._translate_parallel(sentences)
        return " ".join(p for p in parts if p).strip()

    def translate_many(self, texts: list[str]) -> list[str]:
        """Translate several sentences concurrently, aligned with ``texts``.

        The pipeline drains a batch (up to TRANSLATE_MAX_BATCH) and calls this on
        the hot path. Doing one HTTP request per sentence *sequentially* would
        add N× latency; instead we fan the batch out over a small thread pool so
        the batch costs roughly one round-trip. Empty inputs map to ``""``.
        """
        return self._translate_parallel(texts)

    def warmup(self) -> None:
        """Prime the connection + model so the first live segment isn't slow."""
        try:
            self._translate_one("テスト", update_context=False)
        except Exception as exc:  # pragma: no cover - warmup is best-effort
            logger.warning("RouterTranslator warmup failed: %s", exc)

    def close(self) -> None:
        """Release the pooled HTTP connections (call on pipeline shutdown)."""
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 - best-effort
            pass

    # ---- internals --------------------------------------------------------- #

    def _translate_parallel(self, texts: list[str]) -> list[str]:
        """Translate a list concurrently, preserving order; ``""`` for blanks.

        Context updates are disabled inside the fan-out (the worker threads would
        race on the history deque); coherence across a single batch is a minor
        loss since those sentences were spoken at the same time.
        """
        out: list[str] = [""] * len(texts)
        jobs = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not jobs:
            return out
        if len(jobs) == 1:
            i, t = jobs[0]
            out[i] = self._translate_one(t)
            return out
        workers = min(self.max_parallel, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._translate_one, t, False): i for i, t in jobs}
            for fut, i in futures.items():
                try:
                    out[i] = fut.result()
                except Exception as exc:  # noqa: BLE001 - isolate one bad segment
                    logger.warning("Router batch item %d failed: %s", i, exc)
                    out[i] = ""
        return out

    def _translate_one(self, text: str, update_context: bool = True) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        # Fix domain ASR misrecognitions on the SOURCE before translating
        # (post_correct maps Japanese→Japanese, e.g. クロス祖母→クロステナント).
        cleaned = post_correct(cleaned)

        # Pure filler / back-channel: answer without a network round-trip.
        filler = _FILLER_MAP.get(cleaned)
        if filler is not None:
            return filler

        # Deterministic domain substitutions so the model receives clean,
        # unambiguous input (names romanized, loanwords/proper nouns mapped).
        prepared = self._preprocess(cleaned)

        messages = self._build_messages(prepared)
        raw = self._post_with_retry(messages)
        result = self._clean(raw)

        if result and _looks_like_refusal_or_echo(prepared, result):
            logger.debug("Router output rejected (refusal/echo): %r", result[:80])
            result = ""

        if result and update_context and self._keep_context:
            with self._lock:
                self._history.append((cleaned, result))
        return result

    @staticmethod
    def _preprocess(text: str) -> str:
        """Apply deterministic, Latin-only substitutions to the source.

        Only replacements that are plain ASCII (romaji names, English loanwords,
        acronyms) are injected — they copy through a still-Japanese sentence
        cleanly. Vietnamese-valued terms are deliberately NOT substituted here
        (that would create a Japanese/Vietnamese hybrid the model misreads as
        broken input); they are pinned via the system-prompt glossary instead.

        Order matters: names with honorifics first (so 深瀬さん → Fukase-san, not
        a bare "Fukase" that loses the honorific), then bare surnames, katakana
        loanwords, and proper nouns — each longest-first to avoid substring
        corruption. Same shared data the local LLM backend pre-applies.
        """
        out = text
        # Person names + honorific. Surnames <2 chars are skipped for bare
        # replacement to avoid false positives on common single kanji.
        for kanji, romaji in _SORTED_SURNAMES:
            if f"{kanji}さん" in out:
                out = out.replace(f"{kanji}さん", f"{romaji}-san")
            if f"{kanji}様" in out:
                out = out.replace(f"{kanji}様", f"{romaji}-sama")
            if len(kanji) >= 2 and kanji in out:
                out = out.replace(kanji, romaji)
        for kana, romaji in _SORTED_KATAKANA_NAMES:
            if f"{kana}さん" in out:
                out = out.replace(f"{kana}さん", f"{romaji}-san")
            elif kana in out:
                out = out.replace(kana, romaji)
        # Katakana IT/medical loanwords → English; then kanji proper nouns.
        for kana, eng in _SORTED_KATAKANA_TERMS:
            if kana in out:
                out = out.replace(kana, eng)
        for noun, repl in _SORTED_PROPER_NOUNS:
            if noun in out:
                out = out.replace(noun, repl)
        return out

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        # Wrap the Japanese in an XML tag so the model treats it as DATA to
        # translate, not an instruction to obey/answer — a documented defense
        # against weak models slipping into "assistant mode" (OpenAI/Anthropic
        # prompt-eng docs). History src is wrapped the same way for a consistent
        # input→output pattern.
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self._keep_context:
            with self._lock:
                history = list(self._history)
            for src, dst in history:
                messages.append({"role": "user", "content": f"<source_ja>{src}</source_ja>"})
                messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": f"<source_ja>{text}</source_ja>"})
        return messages

    def _post_with_retry(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = self._session.post(self.url, json=body, timeout=self.timeout)
                resp.raise_for_status()
                return self._extract(resp.text)
            except requests.RequestException as exc:
                # Only network/timeout/HTTP errors retry; a malformed-but-200
                # body is handled in _extract (returns "") and never reaches here.
                last_exc = exc
                logger.warning("Router request failed (attempt %d/2): %s", attempt, exc)
        logger.error("Router translation gave up after 2 attempts: %s", last_exc)
        return ""

    @staticmethod
    def _extract(text: str) -> str:
        # The 9router gateway appends a trailing SSE marker ("...}data: [DONE]")
        # to some non-streaming responses, which breaks a plain json.loads /
        # resp.json() with "Extra data". raw_decode reads just the first JSON
        # object and ignores the trailing bytes. Hit on every model (verified).
        try:
            payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
            return str(payload["choices"][0]["message"]["content"] or "")
        except (ValueError, KeyError, IndexError, TypeError):
            logger.warning("Unexpected router response shape: %r", text[:200])
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        """Collapse to a single trimmed line (the pipeline shows one line/segment)."""
        if not text:
            return ""
        line = " ".join(text.split())
        # Strip a leading "VI:" / "Vietnamese:" label some models emit.
        for prefix in ("VI:", "vi:", "Vietnamese:", "Tiếng Việt:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        return line
