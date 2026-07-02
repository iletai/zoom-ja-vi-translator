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
sits between them (です えっ… → split). Greeting set-phrases that carry no
polite terminal (こんにちは / ようこそ / …) are also treated as boundaries so
greeting run-ons (ようこそ皆さんお元気ですか) do not reach NLLB as one block.
"""
from __future__ import annotations


def split_japanese_sentences(text: str) -> list[str]:
    """Split Japanese ``text`` into whole sentences (stateless, pure).

    Reuses :class:`SentenceAggregator`'s tested boundary policy (hard finals
    plus polite terminals guarded by continuation particles) without any
    streaming state. Behaviour:

    - a single sentence returns ``[text]`` (stripped),
    - empty / whitespace-only input returns ``[]``,
    - a multi-sentence string is split at every boundary, including a terminal
      sitting at the very end.

    This is the segmentation NLLB needs: feeding it a multi-sentence block makes
    it silently translate only the first sentence and drop the rest, so the
    translator splits first and translates each sentence in turn.
    """
    if not text or not text.strip():
        return []
    aggregator = SentenceAggregator()
    sentences = aggregator.add(text)
    sentences.extend(aggregator.flush())
    return sentences or [text.strip()]


class SentenceAggregator:
    """Join ASR fragments and emit whole, translatable Japanese sentences."""

    _HARD_FINALS = frozenset("。！？!?")

    # Polite terminal forms, matched longest-first so e.g. ませんでした wins
    # over ません and ですか over です.
    _BASES = (
        "ませんでした",
        "ましょう",
        "でしょう",
        "ください",
        "下さい",
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
        "ということ",
        "っていう",
        "ってば",
        "って感じ",
        "って",
        "みたいな",
        "みたいに",
        "みたいで",
        "みたい",
        "ような",
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

    # Greeting / set-phrase sentence units that carry NO polite terminal and so
    # would otherwise glue to the next turn (e.g. ようこそ皆さん…, こんにちは皆さん…).
    # Streaming JA ASR emits no punctuation, so these run-ons reach NLLB as one
    # block and the trailing sentences are silently dropped. Splitting after a
    # greeting recovers them; over-splitting a genuine single greeting only mildly
    # affects wording, never loses content. Matched longest-first.
    _GREETINGS = (
        "おはようございます",
        "ありがとうございます",
        "はじめまして",
        "こんにちは",
        "こんばんは",
        "おはよう",
        "ようこそ",
    )

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
         "は", "で", "も", "よ", "ね", "かな", "って",
         "ですね", "ますね", "ですよね", "だよね", "ですよ",
         "みたいな", "っぽい", "ということ", "ということで",
         "なんですけど", "なんですが"}
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
            if self.is_dangling(remainder) and sentences and sentences[-1][-1] not in self._HARD_FINALS:
                # At shutdown do not send a low-content tail to NLLB by itself;
                # attach it to the sentence it was trailing so no text is lost.
                # Guard: don't append to a sentence ending with 。！？ — that
                # produces broken Japanese (e.g. "完了しました。ので").
                sentences[-1] += remainder
            else:
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

    # Connective endings that signal an incomplete thought (sentence was cut off).
    _CONNECTIVE_ENDINGS = (
        "だったりで", "たりで", "たりして",
        "ために", "ためで",
        "ので", "のに", "から",
        "けど", "けれど", "けれども",
        "ながら", "つつ", "たり",
        "として", "に対して", "について",
        "によって", "に関して",
        "だけど", "ですが", "ですけど",
    )

    def ends_with_connective(self, text: str) -> bool:
        """True if ``text`` ends with a connective particle (incomplete sentence).

        Unlike :meth:`is_dangling`, this checks the *suffix* of longer text,
        not just exact short-fragment matches.
        """
        text = text.strip()
        if not text:
            return False
        for ending in self._CONNECTIVE_ENDINGS:
            if text.endswith(ending):
                return True
        return False

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

            # Secondary: greeting / set-phrase boundary (no polite terminal).
            for greeting in self._GREETINGS:
                if text.startswith(greeting, index):
                    end = index + len(greeting)
                    rest = text[end:]
                    # Don't split when the greeting is grammatically continued
                    # by a binding particle (こんにちは*と*言う, ようこそ日本*へ*…):
                    # treat as mid-sentence. NOTE: deliberately excludes は/が/も/で
                    # because では/でも/はい/もう commonly START the next sentence
                    # after a greeting (こんにちはでは始めます → must split).
                    if rest and rest[0] in "とのにへを":
                        break
                    continuation_rest = rest[1:] if rest.startswith("、") else rest
                    if any(
                        continuation_rest.startswith(cont)
                        for cont in self._CONTINUATIONS
                    ):
                        break
                    if end < n:
                        return end
                    if flush:
                        return end
                    # Greeting at end of buffer: more may still arrive.
                    break

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
        if base in ("ください", "下さい") and any(
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
                continue
            # よ/わ/さ/ね are ALSO the first kana of very common content words
            # (よろしく, わかりました, さようなら, ねえ…). Treat one as a
            # sentence-final particle only when what follows is genuinely
            # terminal — end of buffer, hard punctuation, whitespace, or a
            # turn-starter; otherwise it begins the next word, so stop and split
            # BEFORE it (at the base boundary) instead of mid-word.
            # EXCEPTION: ね immediately after a polite base (ですね / ますね) is
            # almost always a true sentence-final particle even when the next
            # clause follows directly (ですねとても…), so consume it greedily to
            # avoid orphaning ね onto the next sentence.
            if text[j] == "ね" and j == end:
                j += 1
                continue
            # よ is far more ambiguous: it is also the first kana of よろしく,
            # which is extremely common right after a polite base (ますよろしく).
            # So consume a base-adjacent よ only when what follows is itself a
            # final particle (ですよね → よ then ね); a trailing/terminal よ
            # (ですよ。 / end of buffer) is handled by the generic ``after``
            # check below, and ますよろしく correctly splits BEFORE よ.
            if (
                text[j] == "よ"
                and j == end
                and text[j + 1 : j + 2] in self._FINAL_PARTICLES
            ):
                j += 1
                continue
            after = text[j + 1 :]
            if (
                not after
                or after[0] in self._HARD_FINALS
                or after[0].isspace()
                or any(after.lstrip().startswith(s) for s in self._STARTERS)
            ):
                j += 1
                continue
            break
        # Absorb a hard final (。！？) that closes this same sentence.
        if j < n and text[j] in self._HARD_FINALS:
            j += 1
        return j
