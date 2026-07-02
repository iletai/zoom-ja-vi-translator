"""Local agreement buffering for unstable streaming ASR hypotheses."""
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Tuple


def longest_common_prefix(a: str, b: str) -> str:
    """Return the longest character prefix shared by two strings."""
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return a[:idx]


def _common_prefix_all(values: Iterable[str]) -> str:
    iterator = iter(values)
    try:
        prefix = next(iterator)
    except StopIteration:
        return ""
    for value in iterator:
        prefix = longest_common_prefix(prefix, value)
        if not prefix:
            break
    return prefix


class LocalAgreementBuffer:
    """Commit only characters stable across the last N partial hypotheses.

    The default N=2 is the standard LocalAgreement-2 policy: a character becomes
    committed after it appears unchanged in two consecutive streaming partials.
    """

    def __init__(self, n: int = 2) -> None:
        self.n = max(1, int(n))
        self.committed = ""
        self.prev = ""
        self._history: Deque[str] = deque(maxlen=self.n)

    def reset(self) -> None:
        """Clear all committed and hypothesis history state."""
        self.committed = ""
        self.prev = ""
        self._history.clear()

    def update(self, hypothesis: str) -> Tuple[str, str]:
        """Process a new partial and return ``(committed, volatile_tail)``."""
        h = hypothesis or ""

        if self.n <= 1:
            self.committed = h
            self.prev = h
            self._history.clear()
            self._history.append(h)
            return self.committed, ""

        committed = self.committed
        if committed and not h.startswith(committed):
            committed = longest_common_prefix(committed, h)

        if self.n == 2:
            agreed = longest_common_prefix(self.prev, h)
        else:
            self._history.append(h)
            if len(self._history) >= self.n:
                agreed = _common_prefix_all(self._history)
            else:
                agreed = ""

        if len(agreed) > len(committed):
            committed = agreed
        # committed is already a prefix of h: line 61-62 guarantees this invariant.

        self.committed = committed
        self.prev = h
        return self.committed, h[len(self.committed) :]
