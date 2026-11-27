"""Concurrency caps and per-caller rate limiters (in-memory and optional Redis).

- ``TokenBucketRateLimiter`` protects each individual gateway replica with
  bounded LRU memory usage and zero external dependencies.
- ``RedisRateLimiter`` provides cluster-wide synchronized token-bucket rate
  limiting across multi-replica Kubernetes deployments via an atomic Lua script.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger("project_nexus.rate_limit")


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of evaluating a caller against a rate limiter."""

    allowed: bool
    retry_after_seconds: int
    remaining_tokens: float


class RateLimiter(Protocol):
    """Common interface for in-memory and distributed rate limiters."""

    @property
    def enabled(self) -> bool: ...

    def check(self, caller_id: str) -> RateLimitDecision: ...

    def close(self) -> None: ...


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

    def close(self) -> None:
        return None

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


# Atomic token-bucket Lua script executed server-side in Redis in a single round-trip.
_REDIS_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl_sec = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1])
local last_ts = tonumber(data[2])

if tokens == nil or last_ts == nil then
    tokens = capacity
    last_ts = now
else
    local elapsed = math.max(0, now - last_ts)
    tokens = math.min(capacity, tokens + elapsed * refill_per_sec)
    last_ts = now
end

if tokens >= 1.0 then
    tokens = tokens - 1.0
    redis.call("HMSET", key, "tokens", tokens, "ts", last_ts)
    redis.call("EXPIRE", key, ttl_sec)
    return {1, 0, tostring(tokens)}
else
    local deficit = 1.0 - tokens
    local retry_after = math.max(1, math.ceil(deficit / refill_per_sec))
    redis.call("HMSET", key, "tokens", tokens, "ts", last_ts)
    redis.call("EXPIRE", key, ttl_sec)
    return {0, retry_after, "0"}
end
"""


class RedisRateLimiter:
    """Cluster-wide token-bucket rate limiter backed by Redis Lua execution."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        requests_per_minute: int,
        burst: int | None = None,
        key_prefix: str = "nexus:rl:",
        fail_open: bool = True,
        socket_timeout_seconds: float = 1.0,
        client: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be >= 0")

        self._rpm = requests_per_minute
        self._refill_per_second = requests_per_minute / 60.0
        self._capacity = float(burst if burst is not None else max(1, requests_per_minute))
        self._key_prefix = key_prefix
        self._fail_open = fail_open
        self._clock = clock or time.time
        self._owns_client = client is None

        if client is not None:
            self._client = client
        elif self._rpm > 0:
            if not redis_url:
                raise ValueError("redis_url is required when using RedisRateLimiter")
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional extra missing
                raise RuntimeError(
                    "Redis rate limiting requires the optional 'redis' dependency: "
                    "pip install 'project-nexus[redis]'"
                ) from exc
            self._client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
                decode_responses=True,
            )
        else:
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._rpm > 0

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                close_fn()

    def check(self, caller_id: str) -> RateLimitDecision:
        if not self.enabled or self._client is None:
            return RateLimitDecision(allowed=True, retry_after_seconds=0, remaining_tokens=0.0)

        redis_key = f"{self._key_prefix}{caller_id}"
        ttl_seconds = max(60, int(math.ceil(self._capacity / self._refill_per_second) * 2))
        now = self._clock()
        try:
            raw_result = self._client.eval(
                _REDIS_BUCKET_LUA,
                1,
                redis_key,
                str(self._capacity),
                str(self._refill_per_second),
                f"{now:.6f}",
                str(ttl_seconds),
            )
            allowed = int(raw_result[0]) == 1
            retry_after = int(raw_result[1])
            remaining = round(float(raw_result[2]), 2)
            return RateLimitDecision(
                allowed=allowed,
                retry_after_seconds=retry_after,
                remaining_tokens=remaining,
            )
        except Exception as exc:
            logger.warning(
                "Redis rate limiter check failed; applying configured fail policy",
                extra={"error_type": type(exc).__name__},
            )
            if self._fail_open:
                return RateLimitDecision(allowed=True, retry_after_seconds=0, remaining_tokens=0.0)
            return RateLimitDecision(allowed=False, retry_after_seconds=1, remaining_tokens=0.0)
