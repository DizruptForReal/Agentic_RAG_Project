"""Live progress reporting: what an investigation is doing right now, in plain language.

The loop sets a reporter for the investigation it is running (a context variable, so concurrent
investigations in different threads do not mix). Anything deeper in the stack - the LLM client while it
waits for a rate limit, the structured-output repair - can call `report()` without knowing which
investigation it belongs to.

The board lives in process memory: fine for one API process. With several workers it would move to Redis
(see docs/system_design.md).
"""
from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_reporter: contextvars.ContextVar[Callable[[str, str], None] | None] = contextvars.ContextVar("reporter", default=None)
_wait_acc: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("wait_acc", default=None)


class ActivityBoard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}

    def set(self, inv_id: str, text: str, kind: str) -> None:
        with self._lock:
            self._items[inv_id] = {"text": text, "kind": kind, "since": time.time()}

    def get(self, inv_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(inv_id)
            return dict(item) if item else None

    def clear(self, inv_id: str) -> None:
        with self._lock:
            self._items.pop(inv_id, None)


BOARD = ActivityBoard()


def report(text: str, kind: str = "work") -> None:
    """kind: work | llm | wait | warn"""
    r = _reporter.get()
    if r:
        r(text, kind)


def add_wait(seconds: float) -> None:
    acc = _wait_acc.get()
    if acc is not None:
        acc[0] += seconds


def take_wait() -> float:
    """Seconds spent waiting on rate limits since the last call."""
    acc = _wait_acc.get()
    if acc is None:
        return 0.0
    value, acc[0] = acc[0], 0.0
    return value


@contextmanager
def reporting(inv_id: str, board: ActivityBoard = BOARD) -> Iterator[None]:
    t1 = _reporter.set(lambda text, kind: board.set(inv_id, text, kind))
    t2 = _wait_acc.set([0.0])
    try:
        yield
    finally:
        _reporter.reset(t1)
        _wait_acc.reset(t2)
        board.clear(inv_id)
