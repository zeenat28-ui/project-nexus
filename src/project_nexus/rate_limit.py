"""In-process concurrency cap and bounded per-caller token-bucket rate limiter.

Note: These controls protect each individual gateway replica against overload
and runaway clients. Multi-replica deployments requiring globally synchronized
quotas should pair Nexus with an edge API gateway or shared rate-limit service.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of evaluating a caller against the token bucket."""

    allowed: bool
    retry_after_seconds: int
    remaining_tokens: float


@dataclass
class _BucketState:
    tokens: float
    last_refill: float


class ConcurrencyLimiter:
    """Fail-fast bounded concurrency guard for active upstream inference requests."""

    def __init__(self, *, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._max_concurrent = max_concurrent
        self._active = 0
        self._lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self._max_concurrent:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1


class TokenBucketRateLimiter:
    """Thread-safe per-caller token bucket with bounded LRU memory footprint."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        burst: int | None = None,
        max_tracked_callers: int = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be >= 0")
        if max_tracked_callers < 1:
            raise ValueError("max_tracked_callers must be >= 1")

        self._rpm = requests_per_minute
        self._refill_per_second = requests_per_minute / 60.0
        self._capacity = float(burst if burst is not None else max(1, requests_per_minute))
        self._max_tracked_callers = max_tracked_callers
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, _BucketState] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._rpm > 0

    def tracked_callers(self) -> int:
        with self._lock:
            return len(self._buckets)

    def check(self, caller_id: str) -> RateLimitDecision:
        """Consume one token for ``caller_id`` if available."""
        if not self.enabled:
            return RateLimitDecision(allowed=True, retry_after_seconds=0, remaining_tokens=0.0)

        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(caller_id)
            if bucket is None:
                if len(self._buckets) >= self._max_tracked_callers:
                    self._buckets.popitem(last=False)
                bucket = _BucketState(tokens=self._capacity, last_refill=now)
                self._buckets[caller_id] = bucket
            else:
                self._buckets.move_to_end(caller_id)
                elapsed = max(0.0, now - bucket.last_refill)
                bucket.tokens = min(
                    self._capacity,
                    bucket.tokens + elapsed * self._refill_per_second,
                )
                bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateLimitDecision(
                    allowed=True,
                    retry_after_seconds=0,
                    remaining_tokens=round(bucket.tokens, 2),
                )

            deficit = 1.0 - bucket.tokens
            retry_after = max(1, math.ceil(deficit / self._refill_per_second))
            return RateLimitDecision(
                allowed=False,
                retry_after_seconds=retry_after,
                remaining_tokens=0.0,
            )
