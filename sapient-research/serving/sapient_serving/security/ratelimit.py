"""Per-client rate limiting + daily quota.

Two jobs at once:
  1. Anti-extraction: caps how many (input, output) pairs any single client can
     harvest, which is how you'd try to distill/clone a hosted model. Low limits
     make that slow, expensive, and visible.
  2. Cost guardrail: GPU inference is expensive; quotas bound the bill per client.

This implementation is in-memory (per process). It is correct for a single
instance. For multiple instances / autoscaling, back it with Redis (atomic
INCR + sliding window) so limits are shared across workers. The interface below
stays the same.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, status

from .auth import current_client
from .keys import Client


class _Counter:
    def __init__(self, per_min: int, per_day: int) -> None:
        self.per_min = per_min
        self.per_day = per_day
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = defaultdict(deque)   # client_id -> request timestamps (last 60s)
        self._day: dict[str, tuple[int, int]] = {}                   # client_id -> (utc_day, count)

    def hit(self, client_id: str) -> tuple[bool, str]:
        now = time.time()
        with self._lock:
            # sliding 60s window
            dq = self._recent[client_id]
            cutoff = now - 60.0
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.per_min:
                return False, "rate_limit_per_minute"
            # daily quota
            day = int(now // 86400)
            cur_day, cur_count = self._day.get(client_id, (day, 0))
            if cur_day != day:
                cur_day, cur_count = day, 0
            if cur_count >= self.per_day:
                return False, "daily_quota"
            # record
            dq.append(now)
            self._day[client_id] = (cur_day, cur_count + 1)
            return True, ""


_counter: _Counter | None = None


def init_ratelimit(per_min: int, per_day: int) -> None:
    global _counter
    _counter = _Counter(per_min, per_day)


async def enforce_rate_limit(client: Client = Depends(current_client)) -> Client:
    if _counter is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limiter not initialized")
    ok, reason = _counter.hit(client.client_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({reason}).",
            headers={"Retry-After": "60"},
        )
    return client
