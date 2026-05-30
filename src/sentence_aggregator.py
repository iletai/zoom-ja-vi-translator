"""Pure Japanese sentence aggregation for streaming ASR fragments.

Streaming ASR emits Japanese text WITHOUT punctuation, often gluing several
speaker turns together. Feeding such a run-on string to NLLB makes it silently
drop most of the content. This aggregator re-segments the stream into whole
sentences using a continuation-particle policy:

    split after a polite terminal (です/ます/ました/でしょう/ください/…)
    UNLESS it is immediately followed by a continuation particle
    (が/から/ので/けど/し/ように/って/もの/…)

which keeps mid-sentence forms intact (です*が*, です*から*, ください*まして*…)
while still separating consecutive turns even when no explicit connective word
sits between them (です えっ… → split).
"""
from __future__ import annotations


class SentenceAggregator:
    """Join ASR fragments and emit whole, translatable Japanese sentences."""

    _HARD_FINALS = frozenset("。！？!?.")

    # Polite terminal forms, matched longest-first so e.g. ませんでした wins
    # over ません and ですか over です.
    _BASES = (
        "ませんでした",
        "ましょう",
        "でしょう",
        "ください",
        "でした",
        "ません",
        "ました",
        "です",
        "ます",
    )

    # If a base terminal is immediately followed by one of these, the clause
    # continues (it is NOT a sentence boundary). Longest-first.
    _CONTINUATIONS = (
        "けれども",
        "けれど",
        "けど",
        "という",
        "ってば",
        "って",
        "ように",
        "ので",
        "のに",
        "から",
        "もの",
        "もん",
        "ら",
        "し",
        "と",
        "が",
    )

    # Polite continuations that specifically follow ください (くださいまして…).
    _KUDASAI_SUFFIXES = ("まして", "ません", "ませ", "ます", "まし")

    # Sentence-final particles consumed as part of the terminal before a split.
    _FINAL_PARTICLES = frozenset("かねよわさ")

    # Casual predicate-final の (question/nominalizer) used as a secondary
    # boundary only when a strong turn marker follows.
    _NO_PREDICATES = ("だったの", "たの", "るの", "ないの", "んの")
    _STARTERS = (
        "なるほど",
        "じゃあ",
        "では",
        "でも",
        "ええ",
        "えっ",
        "はい",
        "いや",
        "あの",
        "そう",
        "うん",
    )

    # Low-content dangling fragments that must not be flushed on their own;
    # they are kept buffered to merge with the next utterance instead.
    _DANGLING = frozenset(
        {"かと", "ので", "から", "けど", "けれど", "と", "が", "の", "を", "に",
         "は", "で", "も", "よ", "ね", "かな", "って"}
    )

    def __init__(self) -> None:
        self._buffer = ""

    def add(self, fragment: str) -> list[str]:
        """Append a Japanese ASR fragment and return complete sentences."""
        fragment = fragment.strip()
        if not fragment:
            return []
        self._buffer += fragment
        return self._extract(flush=False)

    def flush(self) -> list[str]:
        """Re-split and return all buffered text, then clear the buffer.

        Unlike :meth:`add`, this also splits a terminal sitting at the very end
        of the buffer (no more text will arrive to continue it).
        """
        sentences = self._extract(flush=True)
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder:
            sentences.append(remainder)
        return sentences

    def pending(self) -> str:
        """Return the current incomplete buffered remainder."""
        return self._buffer

    def reset(self) -> None:
        """Clear all aggregation state."""
        self._buffer = ""

    def is_dangling(self, text: str) -> bool:
        """True if ``text`` is a low-content fragment unsafe to flush alone."""
        return text.strip() in self._DANGLING

    # ─── internals ───────────────────────────────────────────────────────
    def _extract(self, flush: bool) -> list[str]:
        sentences: list[str] = []
        while True:
            boundary = self._find_next_boundary(self._buffer, flush)
            if boundary is None:
                return sentences
            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if sentence and any(ch not in self._HARD_FINALS for ch in sentence):
                sentences.append(sentence)

    def _find_next_boundary(self, text: str, flush: bool) -> int | None:
        n = len(text)
        for index in range(n):
            if text[index] in self._HARD_FINALS:
                return index + 1

            matched_base = None
            for base in self._BASES:
                if text.startswith(base, index):
                    matched_base = base
                    break
            if matched_base is not None:
                boundary = self._boundary_after_base(
                    text, index + len(matched_base), matched_base
                )
                if boundary is not None:
                    if boundary < n:
                        return boundary
                    if flush:
                        return boundary
                    # Terminal at end of buffer: keep buffered, a continuation
                    # particle (が/から/…) might still arrive.

            # Secondary: casual predicate-final の followed by a turn marker.
            for predicate in self._NO_PREDICATES:
                if text.startswith(predicate, index):
                    end = index + len(predicate)
                    if any(text[end:].lstrip().startswith(s) for s in self._STARTERS):
                        return end
        return None

    def _boundary_after_base(self, text: str, end: int, base: str) -> int | None:
        """Return the split index after a base terminal, or None if mid-sentence."""
        rest = text[end:]
        if base == "ください" and any(
            rest.startswith(suffix) for suffix in self._KUDASAI_SUFFIXES
        ):
            return None
        for cont in self._CONTINUATIONS:
            if rest.startswith(cont):
                return None
        # Consume trailing final particles, but treat か as continuation when it
        # begins から / かと / かも / かどうか.
        j = end
        n = len(text)
        while j < n and text[j] in self._FINAL_PARTICLES:
            if text[j] == "か":
                nxt = text[j + 1 : j + 2]
                if nxt in ("ら", "と", "も") or text.startswith("かどうか", j):
                    return None
            j += 1
        # Absorb a hard final (。！？) that closes this same sentence.
        if j < n and text[j] in self._HARD_FINALS:
            j += 1
        return j
