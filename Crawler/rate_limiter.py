"""Rate limiting for outbound network requests.

`RateLimiter` is the seam: callers only ever ask "may I make a request right
now?", so the limiting policy can be swapped without touching crawl logic.
`SlidingWindowRateLimiter` is the only implementation for now.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import deque

WINDOW_SECONDS = 60.0


class RateLimiter(ABC):
    """Decides whether a request may be issued right now."""

    @abstractmethod
    def allow(self) -> bool:
        """Return True if one request may be made immediately.

        Non-blocking. This has *acquire* semantics: a True return means the
        limiter has already recorded the request, so the caller must not
        discard it. On False the caller should back off and retry.

        Implementations must be safe to call from multiple threads, since one
        limiter instance enforces a single global budget for every worker.
        """
        raise NotImplementedError


class SlidingWindowRateLimiter(RateLimiter):
    """Allows at most `requests_per_minute` in any 60-second interval.

    A true sliding window rather than a fixed bucket, so the limit holds over
    *every* 60-second interval and not just aligned ones.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError(f"requests_per_minute must be >= 1, got {requests_per_minute}")

        self.requests_per_minute = requests_per_minute
        self._stamps: deque[float] = deque()
        # One budget shared by every worker, so the read-then-append below has
        # to be atomic: without this, two threads could both observe a
        # not-yet-full window and both append, exceeding the limit.
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """See `RateLimiter.allow`. Thread-safe.

        Checks the size first and only evicts when the window is full. The fast
        path can leave expired stamps in the deque, but that never yields a
        wrong answer: eviction always runs before a False is returned. The
        deque is bounded at `requests_per_minute` entries.
        """
        with self._lock:
            now = time.monotonic()

            if len(self._stamps) < self.requests_per_minute:
                self._stamps.append(now)
                return True

            while self._stamps and now - self._stamps[0] >= WINDOW_SECONDS:
                self._stamps.popleft()

            if len(self._stamps) < self.requests_per_minute:
                self._stamps.append(now)
                return True

            return False
