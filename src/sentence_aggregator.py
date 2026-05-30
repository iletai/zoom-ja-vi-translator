"""Pure Japanese sentence aggregation for streaming ASR fragments."""
from __future__ import annotations


class SentenceAggregator:
    """Join ASR fragments and emit only high-confidence Japanese sentences."""

    _HARD_FINALS = frozenset("。！？!?.")
    _TERMINALS = (
        "ください",
        "ましょう",
        "でしょう",
        "でした",
        "ません",
        "ました",
        "ですか",
        "ますか",
        "ますね",
        "ですね",
        "ますよ",
        "ですよ",
        "です",
        "ます",
    )
    _CLAUSE_STARTERS = (
        "ありがとうございます",
        "ありがとう",
        "それでは",
        "なるほど",
        "それは",
        "じゃあ",
        "では",
        "でも",
        "はい",
        "ええ",
        "そう",
        "さて",
        "ただ",
        "うん",
        "あの",
        "えっと",
        "えー",
    )

    def __init__(self) -> None:
        self._buffer = ""

    def add(self, fragment: str) -> list[str]:
        """Append a Japanese ASR fragment and return complete sentences."""
        fragment = fragment.strip()
        if not fragment:
            return []

        self._buffer += fragment
        return self._extract_complete_sentences()

    def flush(self) -> list[str]:
        """Return the remaining buffered text as a final sentence and clear it."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []

    def pending(self) -> str:
        """Return the current incomplete buffered remainder."""
        return self._buffer

    def reset(self) -> None:
        """Clear all aggregation state."""
        self._buffer = ""

    def _extract_complete_sentences(self) -> list[str]:
        sentences: list[str] = []
        while True:
            boundary = self._find_next_boundary(self._buffer)
            if boundary is None:
                return sentences

            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if sentence:
                sentences.append(sentence)

    def _find_next_boundary(self, text: str) -> int | None:
        for index, char in enumerate(text):
            if char in self._HARD_FINALS:
                return index + 1

            for terminal in self._TERMINALS:
                end = index + len(terminal)
                if not text.startswith(terminal, index):
                    continue
                if self._has_strong_next_clause(text[end:]):
                    return end
                break
        return None

    def _has_strong_next_clause(self, suffix: str) -> bool:
        suffix = suffix.lstrip()
        return any(suffix.startswith(starter) for starter in self._CLAUSE_STARTERS)
