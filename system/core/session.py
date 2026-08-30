"""
THE SYSTEM'S ONE PIECE OF MEMORY.

Until this file existed, `/capture` read an article, sent the text to the
phone, and forgot it. That single fact is why the system could not answer a
follow-up question: by the time the user says "summarise that", nothing
anywhere knows what "that" was.

This holds the last Document per job, keyed by the job id that `/capture`
ALREADY mints and ALREADY returns to the phone.

DELIBERATELY IN MEMORY, NOT ON DISK. One user, one phone, one laptop. A server
restart loses the last article; that costs one re-capture. State it as a
limitation rather than papering over it.

BOUNDED on both axes: TTL and count. Neither bound is measured; both are
design choices, stated as such.

The clock is injectable so expiry can be TESTED rather than slept through.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Optional

from core.config import SESSION_MAX, SESSION_TTL_S
from core.schemas import Document


class SessionStore:
    """job id -> the Document that was read for it."""

    def __init__(self, ttl_s: float = SESSION_TTL_S,
                 max_items: int = SESSION_MAX,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self._clock = clock
        # Value is [stored_at, document, cursor, last_answer_dict]
        self._items: "OrderedDict[str, list]" = OrderedDict()

    def put(self, job: str, document: Document) -> None:
        if not job:
            return
        self.purge()
        existing_answer = self.get_answer(job)
        self._items.pop(job, None)          # re-inserting moves it to newest
        self._items[job] = [self._clock(), document, 0, existing_answer]
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def get(self, job: str) -> Optional[Document]:
        """The Document, or None if never stored or expired.

        A miss and an expiry are the same answer on purpose: the caller's
        response to both is identical — capture again.
        """
        self.purge()
        hit = self._items.get(job or '')
        return hit[1] if hit else None

    def cursor(self, job: str) -> int:
        """How far through the article "next" has walked. 0 = not started."""
        hit = self._items.get(job or '')
        return hit[2] if hit else 0

    def set_cursor(self, job: str, value: int) -> None:
        hit = self._items.get(job or '')
        if hit:
            hit[2] = max(0, int(value))

    def get_answer(self, job: str) -> Optional[dict]:
        hit = self._items.get(job or '')
        return hit[3] if hit and len(hit) > 3 else None

    def set_answer(self, job: str, answer_dict: dict) -> None:
        hit = self._items.get(job or '')
        if hit:
            # Ensure the list is long enough
            while len(hit) <= 3:
                hit.append(None)
            hit[3] = answer_dict

    def age_of(self, job: str) -> Optional[float]:
        hit = self._items.get(job or '')
        return self._clock() - hit[0] if hit else None

    def jobs(self) -> list:
        self.purge()
        return list(reversed(self._items.keys()))

    def purge(self) -> int:
        """Drop expired entries. Returns how many went — a screening step that
        does not say what it dropped is how the third scoring bug hid."""
        if self.ttl_s <= 0:
            return 0
        cutoff = self._clock() - self.ttl_s
        dead = [k for k, v in self._items.items() if v[0] < cutoff]
        for k in dead:
            self._items.pop(k, None)
        return len(dead)

    def drop(self, job: str) -> bool:
        return self._items.pop(job or '', None) is not None

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, job) -> bool:
        return self.get(job) is not None
