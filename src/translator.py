"""NLLB Japanese-to-Vietnamese translation via CTranslate2."""
from __future__ import annotations

import logging
import pathlib

import ctranslate2
from transformers import AutoTokenizer

import config

logger = logging.getLogger(__name__)


class NllbTranslator:
    """Low-latency Japanese-to-Vietnamese translator backed by NLLB/CTranslate2."""

    def __init__(self) -> None:
        ct2_model_dir = pathlib.Path(config.NLLB_CT2_DIR)
        if not ct2_model_dir.exists():
            raise FileNotFoundError(
                f"CTranslate2 NLLB model directory not found: {ct2_model_dir}. "
                "Run scripts/download_models.py to download or convert the model."
            )

        # Load the tokenizer from the local HF cache. Hitting the Hub on every
        # launch costs ~80s of unauthenticated, rate-limited network round-trips
        # (the "sending unauthenticated requests to the HF Hub" stall). config.py
        # enables HF offline mode once models are present, so this resolves to the
        # on-disk cache (<1s). The local_files_only flag is a belt-and-braces guard.
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.NLLB_HF_MODEL,
            src_lang=config.NLLB_SOURCE_LANG,
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
        """Translate Japanese text to Vietnamese."""
        if not text:
            return ""

        text = self._apply_glossary(text)
        token_ids = self.tokenizer.encode(text)
        source_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        target_prefix = [config.NLLB_TARGET_LANG]

        results = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
            # Beam search improves accuracy while anti-repetition knobs prevent live-caption loops.
            beam_size=config.NLLB_BEAM_SIZE,
            repetition_penalty=config.NLLB_REPETITION_PENALTY,
            no_repeat_ngram_size=config.NLLB_NO_REPEAT_NGRAM_SIZE,
            min_decoding_length=config.NLLB_MIN_DECODING_LENGTH,
            max_decoding_length=config.NLLB_MAX_DECODING_LENGTH,
            max_input_length=config.NLLB_MAX_INPUT_LENGTH,
        )

        target_tokens = results[0].hypotheses[0]
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
