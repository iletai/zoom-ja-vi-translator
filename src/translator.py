"""NLLB Japanese-to-Vietnamese translation via CTranslate2."""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import ctranslate2
from transformers import AutoTokenizer

import config
from src.sentence_aggregator import split_japanese_sentences

logger = logging.getLogger(__name__)


def join_translations(sources, translations):
    """Positionally join VI ``translations`` for their JP ``sources``.

    A non-empty source sentence that yields an empty translation is NOT silently
    dropped (the historical data-loss bug): it becomes a visible ``(...)``
    placeholder and is reported back to the caller so the loss can be logged.
    Empty source sentences contribute nothing (no placeholder).

    Returns ``(joined_text, dropped)`` where ``dropped`` is a list of
    ``(index, source_text)`` for each non-empty source with an empty translation.
    """
    parts: list[str] = []
    dropped: list[tuple[int, str]] = []
    for idx, (src, vi) in enumerate(zip(sources, translations)):
        if vi and vi.strip():
            parts.append(vi.strip())
        elif src and src.strip():
            parts.append("(...)")
            dropped.append((idx, src))
    return " ".join(parts), dropped


class NllbTranslator:
    """Low-latency Japanese-to-Vietnamese translator backed by NLLB/CTranslate2."""

    def __init__(self) -> None:
        ct2_model_dir = pathlib.Path(config.NLLB_CT2_DIR)
        if not ct2_model_dir.exists():
            raise FileNotFoundError(
                f"CTranslate2 NLLB model directory not found: {ct2_model_dir}. "
                "Run scripts/download_models.py to download or convert the model."
            )

        # Load the tokenizer. Prefer the local CT2 model directory (which
        # already contains tokenizer.json + sentencepiece.bpe.model from the
        # download script) to avoid any HF Hub network dependency. Fall back to
        # the HF cache for backward compatibility.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(ct2_model_dir),
                src_lang=config.NLLB_SOURCE_LANG,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.NLLB_HF_MODEL,
                src_lang=config.NLLB_SOURCE_LANG,
                revision=config.NLLB_HF_REVISION,
                local_files_only=config.HF_OFFLINE,
            )
        self.translator = ctranslate2.Translator(
            str(ct2_model_dir),
            device="cpu",
            compute_type=config.NLLB_COMPUTE_TYPE,
            inter_threads=config.NLLB_INTER_THREADS,
            intra_threads=config.NLLB_INTRA_THREADS,
        )

        try:
            self.warmup()
        except Exception as exc:  # pragma: no cover - best-effort latency optimization
            logger.warning("NLLB translator warmup failed: %s", exc)

    def translate(self, text: str) -> str:
        """Translate Japanese ``text`` to Vietnamese.

        NLLB is sentence-trained and silently drops trailing sentences when fed
        a multi-sentence block (verified: "本日の会議を始めます。資料を確認してください。"
        translates only the first clause). Following documented MT best practice
        we segment the source into whole sentences first, translate each, and
        rejoin — so no sentence is ever dropped. Single-sentence input is
        unaffected.
        """
        if not text or not text.strip():
            return ""
        if not config.TRANSLATE_SPLIT_SENTENCES:
            return self._translate_one(text)
        sentences = split_japanese_sentences(text)
        if len(sentences) <= 1:
            return self._translate_one(sentences[0] if sentences else text)
        joined, dropped = join_translations(sentences, self.translate_many(sentences))
        for idx, src in dropped:
            logger.warning("Empty VI translation for JP sentence[%d]: %r", idx, src)
        return joined

    def translate_many(self, texts: list[str]) -> list[str]:
        """Translate several Japanese sentences in one CTranslate2 batch.

        Returns one Vietnamese string per input, positionally aligned (empty
        inputs map to ``""``). Batching keeps a single native call so translating
        N sentences costs far less than N separate calls — the key to keeping up
        with a fast meeting and not building the backlog that causes drops.
        """
        prepared: list[Optional[list[str]]] = []
        batch_tokens: list[list[str]] = []
        for text in texts:
            cleaned = text.strip() if text else ""
            if not cleaned:
                prepared.append(None)
                continue
            cleaned = self._apply_glossary(cleaned)
            token_ids = self.tokenizer.encode(cleaned)
            source_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            prepared.append(source_tokens)
            batch_tokens.append(source_tokens)

        decoded: list[str] = []
        if batch_tokens:
            results = self.translator.translate_batch(
                batch_tokens,
                target_prefix=[[config.NLLB_TARGET_LANG]] * len(batch_tokens),
                beam_size=config.NLLB_BEAM_SIZE,
                repetition_penalty=config.NLLB_REPETITION_PENALTY,
                no_repeat_ngram_size=config.NLLB_NO_REPEAT_NGRAM_SIZE,
                min_decoding_length=config.NLLB_MIN_DECODING_LENGTH,
                max_decoding_length=config.NLLB_MAX_DECODING_LENGTH,
                max_input_length=config.NLLB_MAX_INPUT_LENGTH,
            )
            decoded = [self._decode(result) for result in results]

        out: list[str] = []
        cursor = 0
        for source_tokens in prepared:
            if source_tokens is None:
                out.append("")
            else:
                out.append(decoded[cursor])
                cursor += 1
        return out

    def _translate_one(self, text: str) -> str:
        """Translate a single sentence (or arbitrary string) as one sequence."""
        if not text or not text.strip():
            return ""
        return self.translate_many([text])[0]

    def _decode(self, result) -> str:
        target_tokens = result.hypotheses[0]
        # CTranslate2 includes the forced target language prefix in the output.
        if target_tokens and target_tokens[0] == config.NLLB_TARGET_LANG:
            target_tokens = target_tokens[1:]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        return self.tokenizer.decode(target_ids)

    def warmup(self) -> None:
        """Run one tiny translation to avoid first-call lag."""
        self.translate("テスト")

    @staticmethod
    def _apply_glossary(text: str) -> str:
        """Substitute domain terms NLLB mistranslates with Latin renderings.

        The replacement happens on the Japanese source; NLLB copies the Latin
        text through to the Vietnamese output reliably.
        """
        for term, replacement in config.NLLB_GLOSSARY.items():
            if term in text:
                text = text.replace(term, replacement)
        return text


if __name__ == "__main__":
    if pathlib.Path(config.NLLB_CT2_DIR).exists():
        sample_jp = "本日の会議を始めます。"
        translated_vi = NllbTranslator().translate(sample_jp)
        print(f"JP: {sample_jp}")
        print(f"VI: {translated_vi}")
    else:
        print(
            f"CTranslate2 NLLB model directory not found: {config.NLLB_CT2_DIR}. "
            "Run scripts/download_models.py first."
        )
