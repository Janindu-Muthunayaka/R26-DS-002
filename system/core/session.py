"""
THE SYSTEM'S ONE PIECE OF MEMORY.

Until this file existed, `/capture` read an article, sent the text to the
phone, and forgot it. That single fact is why the system could not answer a
follow-up question: by the time the user says "summarise that", nothing
anywhere knows what "that" was.

This holds the last Document per job, keyed by the job id that `/capture`
ALREADY mints and ALREADY returns to the phone. Nothing new has to be
invented on the wire — the phone keeps the id it is given and sends it back
with the question.

DELIBERATELY IN MEMORY, NOT ON DISK
-----------------------------------
  * One user, one phone, one laptop. There is no multi-user story here to get
    wrong, and pretending otherwise would add a failure mode for nothing.
  * A server restart loses the last article. That costs one re-capture, which
    is four seconds of the user's time. Persisting it buys nothing.

State that as a limitation. Do not describe this as durable storage.

BOUNDED ON BOTH AXES so a long session cannot grow without limit:
  * TTL   — an article nobody asked about within `ttl_s` is dropped.
  * COUNT — at most `max_items`; the oldest is evicted first.

Neither bound is a measured value. Both are design choices, stated as such.

The clock is injectable so the expiry behaviour can be TESTED rather than
slept through — a test that sleeps for thirty minutes is a test nobody runs.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Optional

from core.config import SESSION_MAX, SESSION_TTL_S
from core.schemas import Document


class SessionStore:
    """job id -> the Document that was read for it."""

    def __init__(self,
                 ttl_s: float = SESSION_TTL_S,
                 max_items: int = SESSION_MAX,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self._clock = clock
        # OrderedDict, oldest first. put() appends, so eviction is popitem(last=False).
        # Value is (stored_at, document, cursor). The cursor is how far through
        # the article the listener has walked with "next" - see
        # layers/l6_generator. It lives here rather than in a second dict so
        # that it expires and is evicted WITH its article; a cursor that
        # outlives the text it indexes into is a bug waiting for a demo.
        self._items: "OrderedDict[str, list]" = OrderedDict()

    # ---- writing ---------------------------------------------------------
    def put(self, job: str, document: Document) -> None:
        if not job:
            return
        self.purge()
        self._items.pop(job, None)          # re-inserting moves it to newest
        self._items[job] = [self._clock(), document, 0]
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    # ---- reading ---------------------------------------------------------
    def get(self, job: str) -> Optional[Document]:
        """The Document, or None if it was never stored or has expired.

        A miss and an expiry are the same answer on purpose: the caller's
        response to both is identical — tell the user nothing has been read
        yet and ask them to capture again.
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

    def age_of(self, job: str) -> Optional[float]:
        """Seconds since `job` was stored, or None. Diagnostics only."""
        hit = self._items.get(job or '')
        return self._clock() - hit[0] if hit else None

    def jobs(self) -> list:
        """Live job ids, newest first."""
        self.purge()
        return list(reversed(self._items.keys()))

    # ---- housekeeping ----------------------------------------------------
    def purge(self) -> int:
        """Drop expired entries. Returns how many went — a screening step
        that does not say what it dropped is how the third scoring bug hid."""
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
