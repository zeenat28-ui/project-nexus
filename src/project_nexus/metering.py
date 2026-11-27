"""Multi-tenant usage metering, token/cost quota enforcement, and SQLite/Redis ledgers."""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("project_nexus.metering")

_REDIS_USAGE_RECORD_LUA = """
local key = KEYS[1]
local tenants_key = KEYS[2]
local tenant_id = ARGV[1]
local prompt_t = tonumber(ARGV[2])
local completion_t = tonumber(ARGV[3])
local total_t = tonumber(ARGV[4])
local cost_usd = tonumber(ARGV[5])
local ttl_sec = tonumber(ARGV[6])

redis.call("HINCRBY", key, "requests", 1)
redis.call("HINCRBY", key, "prompt_tokens", prompt_t)
redis.call("HINCRBY", key, "completion_tokens", completion_t)
redis.call("HINCRBY", key, "total_tokens", total_t)
redis.call("HINCRBYFLOAT", key, "estimated_cost_usd", cost_usd)
redis.call("EXPIRE", key, ttl_sec)
redis.call("SADD", tenants_key, tenant_id)
redis.call("EXPIRE", tenants_key, ttl_sec)
return 1
"""


@dataclass(frozen=True)
class ModelPricing:
    """Per-1K token USD pricing for a model."""

    prompt_cost_per_1k_usd: float = 0.0
    completion_cost_per_1k_usd: float = 0.0


@dataclass(frozen=True)
class TenantBudget:
    """Per-window token and cost ceiling for a tenant."""

    max_tokens_per_window: int | None = None
    max_cost_usd_per_window: float | None = None


@dataclass(frozen=True)
class QuotaDecision:
    """Result of checking whether a tenant has remaining budget."""

    allowed: bool
    reason: str | None = None
    retry_after_seconds: int = 0
    tokens_used: int = 0
    cost_usd_used: float = 0.0


@dataclass(frozen=True)
class UsageRecordResult:
    """Recorded usage and estimated cost for a single completed request."""

    tenant_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class _WindowAccumulator:
    window_id: int
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


def estimate_tokens_from_text(text: str) -> int:
    """Deterministic fallback token estimator (~4 chars/token) when backend omits usage."""
    stripped = text.strip()
    if not stripped:
        return 0
    by_words = len(stripped.split())
    by_chars = math.ceil(len(stripped) / 4)
    return max(1, by_words, by_chars)


class UsageMeter:
    """Thread-safe multi-tenant token metering, cost accounting, and quota enforcement.

    Supports three storage modes:
    - in-memory (default, single replica)
    - SQLite (single replica durable persistence via ``sqlite_path``)
    - Redis (cluster-wide synchronized metering across HA replicas via
      ``redis_url`` / ``redis_client``)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_seconds: int = 86_400,
        default_pricing: ModelPricing | None = None,
        model_pricing: Mapping[str, ModelPricing] | None = None,
        default_budget: TenantBudget | None = None,
        tenant_budgets: Mapping[str, TenantBudget] | None = None,
        sqlite_path: str | Path | None = None,
        redis_url: str | None = None,
        redis_client: Any | None = None,
        redis_key_prefix: str = "nexus:usage:",
        redis_fail_open: bool = True,
        redis_timeout_seconds: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_seconds < 60:
            raise ValueError("window_seconds must be >= 60")

        self._enabled = enabled
        self._window_seconds = window_seconds
        self._default_pricing = default_pricing or ModelPricing()
        self._model_pricing: dict[str, ModelPricing] = dict(model_pricing or {})
        self._default_budget = default_budget or TenantBudget()
        self._tenant_budgets: dict[str, TenantBudget] = dict(tenant_budgets or {})
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._windows: dict[str, _WindowAccumulator] = {}
        self._conn: sqlite3.Connection | None = None
        self._redis: Any | None = None
        self._owns_redis = False
        self._redis_prefix = redis_key_prefix
        self._redis_fail_open = redis_fail_open

        if redis_client is not None:
            self._redis = redis_client
        elif redis_url is not None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional extra missing
                raise RuntimeError(
                    "Redis metering requires the optional 'redis' extra: "
                    "pip install 'project-nexus[redis]'"
                ) from exc
            self._redis = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=redis_timeout_seconds,
                socket_timeout=redis_timeout_seconds,
                decode_responses=True,
            )
            self._owns_redis = True
        elif sqlite_path is not None:
            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._init_sqlite()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def window_seconds(self) -> int:
        return self._window_seconds

    @property
    def storage_backend(self) -> str:
        if self._redis is not None:
            return "redis"
        if self._conn is not None:
            return "sqlite"
        return "memory"

    def _init_sqlite(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nexus_usage_windows (
                tenant_id TEXT NOT NULL,
                window_id INTEGER NOT NULL,
                requests INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                PRIMARY KEY (tenant_id, window_id)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._owns_redis and self._redis is not None:
                close_fn = getattr(self._redis, "close", None)
                if callable(close_fn):
                    close_fn()
                self._redis = None

    def _current_window(self) -> tuple[int, int]:
        now = self._clock()
        window_id = int(now // self._window_seconds)
        window_end = (window_id + 1) * self._window_seconds
        retry_after = max(1, math.ceil(window_end - now))
        return window_id, retry_after

    def _redis_keys(self, tenant_id: str, window_id: int) -> tuple[str, str]:
        return (
            f"{self._redis_prefix}{window_id}:{tenant_id}",
            f"{self._redis_prefix}{window_id}:tenants",
        )

    def _get_budget(self, tenant_id: str) -> TenantBudget:
        return self._tenant_budgets.get(tenant_id, self._default_budget)

    def _get_pricing(self, model: str) -> ModelPricing:
        return self._model_pricing.get(model, self._default_pricing)

    def _read_redis_window(self, tenant_id: str, window_id: int) -> _WindowAccumulator:
        assert self._redis is not None
        usage_key, _ = self._redis_keys(tenant_id, window_id)
        raw = self._redis.hgetall(usage_key) or {}
        return _WindowAccumulator(
            window_id=window_id,
            requests=int(raw.get("requests", 0)),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
            total_tokens=int(raw.get("total_tokens", 0)),
            estimated_cost_usd=float(raw.get("estimated_cost_usd", 0.0)),
        )

    def _load_accumulator_locked(self, tenant_id: str, window_id: int) -> _WindowAccumulator:
        acc = self._windows.get(tenant_id)
        if acc is not None and acc.window_id == window_id:
            return acc

        if self._conn is not None:
            row = self._conn.execute(
                """
                SELECT requests, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd
                FROM nexus_usage_windows
                WHERE tenant_id = ? AND window_id = ?
                """,
                (tenant_id, window_id),
            ).fetchone()
            if row is not None:
                acc = _WindowAccumulator(
                    window_id=window_id,
                    requests=int(row[0]),
                    prompt_tokens=int(row[1]),
                    completion_tokens=int(row[2]),
                    total_tokens=int(row[3]),
                    estimated_cost_usd=float(row[4]),
                )
                self._windows[tenant_id] = acc
                return acc

        acc = _WindowAccumulator(window_id=window_id)
        self._windows[tenant_id] = acc
        return acc

    def _persist_sqlite_locked(self, tenant_id: str, acc: _WindowAccumulator) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT INTO nexus_usage_windows (
                tenant_id, window_id, requests, prompt_tokens, completion_tokens,
                total_tokens, estimated_cost_usd
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, window_id) DO UPDATE SET
                requests = excluded.requests,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens = excluded.total_tokens,
                estimated_cost_usd = excluded.estimated_cost_usd
            """,
            (
                tenant_id,
                acc.window_id,
                acc.requests,
                acc.prompt_tokens,
                acc.completion_tokens,
                acc.total_tokens,
                acc.estimated_cost_usd,
            ),
        )
        self._conn.commit()

    def check_quota(self, tenant_id: str) -> QuotaDecision:
        """Verify whether ``tenant_id`` has remaining token and cost budget."""
        if not self._enabled:
            return QuotaDecision(allowed=True)

        budget = self._get_budget(tenant_id)
        if budget.max_tokens_per_window is None and budget.max_cost_usd_per_window is None:
            return QuotaDecision(allowed=True)

        window_id, retry_after = self._current_window()
        if self._redis is not None:
            try:
                acc = self._read_redis_window(tenant_id, window_id)
            except Exception as exc:
                logger.warning(
                    "Redis metering quota check failed; applying fail policy",
                    extra={"error_type": type(exc).__name__},
                )
                if self._redis_fail_open:
                    return QuotaDecision(allowed=True)
                return QuotaDecision(
                    allowed=False,
                    reason="token_budget",
                    retry_after_seconds=1,
                )
        else:
            with self._lock:
                acc = self._load_accumulator_locked(tenant_id, window_id)

        if (
            budget.max_tokens_per_window is not None
            and acc.total_tokens >= budget.max_tokens_per_window
        ):
            return QuotaDecision(
                allowed=False,
                reason="token_budget",
                retry_after_seconds=retry_after,
                tokens_used=acc.total_tokens,
                cost_usd_used=round(acc.estimated_cost_usd, 6),
            )
        if (
            budget.max_cost_usd_per_window is not None
            and acc.estimated_cost_usd >= budget.max_cost_usd_per_window
        ):
            return QuotaDecision(
                allowed=False,
                reason="cost_budget",
                retry_after_seconds=retry_after,
                tokens_used=acc.total_tokens,
                cost_usd_used=round(acc.estimated_cost_usd, 6),
            )
        return QuotaDecision(
            allowed=True,
            tokens_used=acc.total_tokens,
            cost_usd_used=round(acc.estimated_cost_usd, 6),
        )

    def record_usage(
        self,
        *,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageRecordResult:
        """Record completed request token usage and computed USD cost."""
        safe_prompt = max(0, prompt_tokens)
        safe_completion = max(0, completion_tokens)
        total = safe_prompt + safe_completion

        pricing = self._get_pricing(model)
        cost_usd = (safe_prompt / 1000.0) * pricing.prompt_cost_per_1k_usd + (
            safe_completion / 1000.0
        ) * pricing.completion_cost_per_1k_usd

        if self._enabled:
            window_id, _ = self._current_window()
            if self._redis is not None:
                usage_key, tenants_key = self._redis_keys(tenant_id, window_id)
                ttl_seconds = self._window_seconds * 2
                try:
                    self._redis.eval(
                        _REDIS_USAGE_RECORD_LUA,
                        2,
                        usage_key,
                        tenants_key,
                        tenant_id,
                        str(safe_prompt),
                        str(safe_completion),
                        str(total),
                        f"{cost_usd:.10f}",
                        str(ttl_seconds),
                    )
                except Exception as exc:
                    logger.warning(
                        "Redis metering usage write failed",
                        extra={"error_type": type(exc).__name__},
                    )
            else:
                with self._lock:
                    acc = self._load_accumulator_locked(tenant_id, window_id)
                    acc.requests += 1
                    acc.prompt_tokens += safe_prompt
                    acc.completion_tokens += safe_completion
                    acc.total_tokens += total
                    acc.estimated_cost_usd += cost_usd
                    self._persist_sqlite_locked(tenant_id, acc)

        return UsageRecordResult(
            tenant_id=tenant_id,
            prompt_tokens=safe_prompt,
            completion_tokens=safe_completion,
            total_tokens=total,
            estimated_cost_usd=round(cost_usd, 8),
        )

    def get_tenant_summary(self, tenant_id: str) -> dict[str, Any]:
        """Return current-window usage and remaining budget for one tenant."""
        window_id, reset_in = self._current_window()
        budget = self._get_budget(tenant_id)
        if self._redis is not None:
            try:
                acc = self._read_redis_window(tenant_id, window_id)
            except Exception:
                acc = _WindowAccumulator(window_id=window_id)
        else:
            with self._lock:
                acc = self._load_accumulator_locked(tenant_id, window_id)

        tokens_remaining = (
            max(0, budget.max_tokens_per_window - acc.total_tokens)
            if budget.max_tokens_per_window is not None
            else None
        )
        cost_remaining = (
            round(max(0.0, budget.max_cost_usd_per_window - acc.estimated_cost_usd), 6)
            if budget.max_cost_usd_per_window is not None
            else None
        )
        return {
            "tenant_id": tenant_id,
            "storage_backend": self.storage_backend,
            "window_seconds": self._window_seconds,
            "window_resets_in_seconds": reset_in,
            "requests": acc.requests,
            "prompt_tokens": acc.prompt_tokens,
            "completion_tokens": acc.completion_tokens,
            "total_tokens": acc.total_tokens,
            "estimated_cost_usd": round(acc.estimated_cost_usd, 6),
            "limits": {
                "max_tokens_per_window": budget.max_tokens_per_window,
                "remaining_tokens": tokens_remaining,
                "max_cost_usd_per_window": budget.max_cost_usd_per_window,
                "remaining_cost_usd": cost_remaining,
            },
        }

    def get_all_summaries(self) -> dict[str, Any]:
        """Return org-wide current-window usage across all active tenants."""
        window_id, reset_in = self._current_window()
        tenant_ids: set[str] = set(self._tenant_budgets.keys())
        if self._redis is not None:
            _, tenants_key = self._redis_keys("_", window_id)
            try:
                members = self._redis.smembers(tenants_key) or set()
                tenant_ids.update(str(m) for m in members)
            except Exception:  # noqa: S110 - return configured budgets on Redis outage
                pass
        else:
            with self._lock:
                tenant_ids.update(self._windows.keys())
                if self._conn is not None:
                    rows = self._conn.execute(
                        "SELECT DISTINCT tenant_id FROM nexus_usage_windows WHERE window_id = ?",
                        (window_id,),
                    ).fetchall()
                    tenant_ids.update(str(row[0]) for row in rows)

        tenants_map = {tid: self.get_tenant_summary(tid) for tid in sorted(tenant_ids)}
        total_requests = sum(int(item["requests"]) for item in tenants_map.values())
        total_tokens = sum(int(item["total_tokens"]) for item in tenants_map.values())
        total_cost = round(
            sum(float(item["estimated_cost_usd"]) for item in tenants_map.values()), 6
        )
        return {
            "storage_backend": self.storage_backend,
            "window_seconds": self._window_seconds,
            "window_resets_in_seconds": reset_in,
            "totals": {
                "requests": total_requests,
                "total_tokens": total_tokens,
                "estimated_cost_usd": total_cost,
            },
            "tenants": tenants_map,
        }
