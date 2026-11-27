"""Heterogeneous multi-backend router with hardware-aware scheduling and KV-cache spillover."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from project_nexus.backends.base import InferenceBackend
from project_nexus.backends.factory import create_backend
from project_nexus.config import AcceleratorType, NexusSettings, RoutingStrategy
from project_nexus.resilience import CircuitBreaker, CircuitState
from project_nexus.telemetry import EngineTelemetrySnapshot
from project_nexus.tracing import NexusTracer


@dataclass
class BackendPoolStats:
    """Mutable per-pool runtime load, latency EWMA, and upstream KV-cache state."""

    in_flight: int = 0
    ewma_latency_ms: float | None = None
    total_requests: int = 0
    telemetry: EngineTelemetrySnapshot | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def enter(self) -> int:
        with self._lock:
            self.in_flight += 1
            return self.in_flight

    def leave(self) -> int:
        with self._lock:
            if self.in_flight > 0:
                self.in_flight -= 1
            return self.in_flight

    def record_latency(self, latency_ms: float, *, alpha: float = 0.25) -> float:
        with self._lock:
            self.total_requests += 1
            if self.ewma_latency_ms is None:
                self.ewma_latency_ms = latency_ms
            else:
                self.ewma_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.ewma_latency_ms
            return self.ewma_latency_ms

    def update_telemetry(self, snapshot: EngineTelemetrySnapshot) -> None:
        with self._lock:
            self.telemetry = snapshot

    def snapshot(self) -> tuple[int, float | None, int, EngineTelemetrySnapshot | None]:
        with self._lock:
            return self.in_flight, self.ewma_latency_ms, self.total_requests, self.telemetry


@dataclass(frozen=True)
class RoutedBackendEntry:
    """Single backend pool paired with hardware metadata, circuit breaker, and live stats."""

    route_name: str
    backend: InferenceBackend
    circuit_breaker: CircuitBreaker
    accelerator: AcceleratorType = "unknown"
    gpu_count: int = 0
    weight: int = 1
    cost_tier: int = 1
    max_kv_cache_usage_ratio: float = 0.95
    stats: BackendPoolStats = field(default_factory=BackendPoolStats)

    def is_kv_saturated(self) -> bool:
        _, _, _, tel = self.stats.snapshot()
        if tel is None or tel.kv_cache_usage_ratio is None:
            return False
        return tel.kv_cache_usage_ratio >= self.max_kv_cache_usage_ratio


class BackendRouter:
    """Routes requests across heterogeneous CUDA/ROCm/CPU pools with load-aware scheduling.

    Each backend pool maintains an isolated ``CircuitBreaker``, live in-flight count,
    EWMA latency estimate, and optional scraped upstream GPU KV-cache utilization.
    """

    def __init__(
        self,
        entries: Mapping[str, RoutedBackendEntry],
        *,
        default_route: str,
        model_routes: Mapping[str, str | list[str]] | None = None,
        strategy: RoutingStrategy = "priority",
        enable_failover: bool = False,
        fallback_order: tuple[str, ...] = (),
        multi_backend: bool = False,
    ) -> None:
        if not entries:
            raise ValueError("BackendRouter requires at least one backend entry.")
        if default_route not in entries:
            raise ValueError(f"default_route {default_route!r} not found in entries.")

        self._entries: dict[str, RoutedBackendEntry] = dict(entries)
        self._default_route = default_route
        self._model_routes: dict[str, tuple[str, ...]] = {}
        for model_id, target in (model_routes or {}).items():
            self._model_routes[model_id] = (target,) if isinstance(target, str) else tuple(target)
        self._strategy: RoutingStrategy = strategy
        self._enable_failover = enable_failover
        self._fallback_order = fallback_order
        self._multi_backend = multi_backend
        self._rr_lock = threading.Lock()
        self._rr_weights: dict[str, int] = dict.fromkeys(self._entries, 0)

    @classmethod
    def single(
        cls,
        backend: InferenceBackend,
        *,
        circuit_breaker: CircuitBreaker,
        accelerator: AcceleratorType = "unknown",
        gpu_count: int = 0,
    ) -> BackendRouter:
        route_name = backend.name
        entry = RoutedBackendEntry(
            route_name=route_name,
            backend=backend,
            circuit_breaker=circuit_breaker,
            accelerator=accelerator,
            gpu_count=gpu_count,
        )
        return cls(
            {route_name: entry},
            default_route=route_name,
            multi_backend=False,
        )

    @classmethod
    def from_settings(
        cls,
        settings: NexusSettings,
        *,
        tracer: NexusTracer | None = None,
    ) -> BackendRouter:
        if not settings.backends:
            backend = create_backend(settings.backend, tracer=tracer)
            breaker = CircuitBreaker(
                failure_threshold=settings.circuit_breaker.failure_threshold,
                cooldown_seconds=settings.circuit_breaker.cooldown_seconds,
            )
            return cls.single(
                backend,
                circuit_breaker=breaker,
                accelerator=settings.backend.accelerator,
                gpu_count=settings.backend.gpu_count,
            )

        entries: dict[str, RoutedBackendEntry] = {}
        for pool_name, pool_cfg in settings.backends.items():
            backend = create_backend(pool_cfg, tracer=tracer)
            breaker = CircuitBreaker(
                failure_threshold=settings.circuit_breaker.failure_threshold,
                cooldown_seconds=settings.circuit_breaker.cooldown_seconds,
            )
            entries[pool_name] = RoutedBackendEntry(
                route_name=pool_name,
                backend=backend,
                circuit_breaker=breaker,
                accelerator=pool_cfg.accelerator,
                gpu_count=pool_cfg.gpu_count,
                weight=pool_cfg.weight,
                cost_tier=pool_cfg.cost_tier,
                max_kv_cache_usage_ratio=pool_cfg.max_kv_cache_usage_ratio,
            )

        default_route = settings.routing.default_backend or next(iter(entries))
        return cls(
            entries,
            default_route=default_route,
            model_routes=settings.routing.model_routes,
            strategy=settings.routing.strategy,
            enable_failover=settings.routing.enable_failover,
            fallback_order=tuple(settings.routing.fallback_order),
            multi_backend=True,
        )

    @property
    def is_multi_backend(self) -> bool:
        return self._multi_backend

    @property
    def strategy(self) -> RoutingStrategy:
        return self._strategy

    @property
    def default_route(self) -> str:
        return self._default_route

    @property
    def default_entry(self) -> RoutedBackendEntry:
        return self._entries[self._default_route]

    def entries(self) -> list[RoutedBackendEntry]:
        return list(self._entries.values())

    def get(self, route_name: str) -> RoutedBackendEntry:
        return self._entries[route_name]

    async def startup(self) -> None:
        await asyncio.gather(*(entry.backend.startup() for entry in self._entries.values()))

    async def shutdown(self) -> None:
        await asyncio.gather(*(entry.backend.shutdown() for entry in self._entries.values()))

    def _order_group(self, pool_names: list[str]) -> list[str]:
        if len(pool_names) <= 1:
            return pool_names

        # Deprioritize pools whose scraped GPU KV-cache has reached saturation threshold.
        non_saturated = [n for n in pool_names if not self._entries[n].is_kv_saturated()]
        saturated = [n for n in pool_names if self._entries[n].is_kv_saturated()]

        def sort_subset(names: list[str]) -> list[str]:
            if len(names) <= 1:
                return names
            if self._strategy == "priority":
                return names
            if self._strategy == "least_inflight":
                return sorted(
                    names,
                    key=lambda n: (
                        self._entries[n].stats.snapshot()[0],
                        self._entries[n].stats.snapshot()[1] or 0.0,
                    ),
                )
            if self._strategy == "latency_ewma":
                return sorted(
                    names,
                    key=lambda n: (
                        -1.0
                        if self._entries[n].stats.snapshot()[1] is None
                        else float(self._entries[n].stats.snapshot()[1] or 0.0)
                    ),
                )
            if self._strategy == "cost_aware":
                return sorted(
                    names,
                    key=lambda n: (
                        self._entries[n].cost_tier,
                        self._entries[n].stats.snapshot()[0],
                    ),
                )
            if self._strategy == "weighted":
                return self._smooth_weighted_order(names)
            return names

        return sort_subset(non_saturated) + sort_subset(saturated)

    def _smooth_weighted_order(self, names: list[str]) -> list[str]:
        """Nginx-style smooth weighted round-robin picking the primary pool deterministically."""
        with self._rr_lock:
            total_weight = sum(self._entries[n].weight for n in names)
            best_name = names[0]
            best_score = -(10**9)
            for name in names:
                self._rr_weights[name] = self._rr_weights.get(name, 0) + self._entries[name].weight
                if self._rr_weights[name] > best_score:
                    best_score = self._rr_weights[name]
                    best_name = name
            self._rr_weights[best_name] -= total_weight
        return [best_name, *[n for n in names if n != best_name]]

    def resolve_candidates(self, requested_model: str | None) -> list[RoutedBackendEntry]:
        """Return ordered backend candidates according to model routes and strategy."""
        primary_group: list[str] = []
        if requested_model:
            if requested_model in self._model_routes:
                primary_group = list(self._model_routes[requested_model])
            else:
                primary_group = [
                    name
                    for name, entry in self._entries.items()
                    if entry.backend.default_model == requested_model
                ]

        if not primary_group:
            primary_group = [self._default_route]

        ordered_names = self._order_group(primary_group)

        if self._enable_failover:
            fallback_candidates = self._fallback_order or tuple(self._entries.keys())
            extra = [
                candidate
                for candidate in fallback_candidates
                if candidate in self._entries and candidate not in ordered_names
            ]
            ordered_names.extend(self._order_group(extra))

        return [self._entries[name] for name in ordered_names]

    def list_models(self) -> list[dict[str, str]]:
        """List all configured models across registered backend pools."""
        seen: set[str] = set()
        models: list[dict[str, str]] = []

        for model_id, pool_names in self._model_routes.items():
            if model_id in seen:
                continue
            seen.add(model_id)
            owner = ",".join(pool_names)
            models.append({"id": model_id, "object": "model", "owned_by": owner})

        for entry in self._entries.values():
            model_id = entry.backend.default_model
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": entry.route_name if self._multi_backend else entry.backend.name,
                }
            )

        return models

    def fleet_status(self) -> dict[str, Any]:
        """Return live per-pool hardware, scheduling, circuit, and KV-cache telemetry."""
        pools: dict[str, dict[str, Any]] = {}
        for entry in self._entries.values():
            snap = entry.circuit_breaker.snapshot()
            in_flight, ewma_ms, total_reqs, tel = entry.stats.snapshot()
            pools[entry.route_name] = {
                "backend_type": entry.backend.name,
                "default_model": entry.backend.default_model,
                "accelerator": entry.accelerator,
                "gpu_count": entry.gpu_count,
                "weight": entry.weight,
                "cost_tier": entry.cost_tier,
                "circuit_state": snap.state.value,
                "in_flight_requests": in_flight,
                "total_requests": total_reqs,
                "ewma_latency_ms": round(ewma_ms, 2) if ewma_ms is not None else None,
                "kv_cache_usage_ratio": tel.kv_cache_usage_ratio if tel else None,
                "queue_depth": tel.queue_depth if tel else None,
                "kv_saturated": entry.is_kv_saturated(),
                "max_kv_cache_usage_ratio": entry.max_kv_cache_usage_ratio,
            }
        return {
            "strategy": self._strategy,
            "default_backend": self._default_route,
            "enable_failover": self._enable_failover,
            "pools": pools,
        }

    async def check_all_health(
        self,
        *,
        timeout_seconds: float,
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Probe health and optional upstream GPU telemetry for every pool concurrently."""

        async def _probe_one(entry: RoutedBackendEntry) -> tuple[str, bool, dict[str, Any]]:
            snap = entry.circuit_breaker.snapshot()
            tel = await entry.backend.scrape_telemetry(timeout=timeout_seconds)
            if tel is not None:
                entry.stats.update_telemetry(tel)

            if snap.state is CircuitState.OPEN:
                return (
                    entry.route_name,
                    False,
                    {
                        "healthy": False,
                        "backend": entry.backend.name,
                        "accelerator": entry.accelerator,
                        "circuit_state": snap.state.value,
                        "retry_after_seconds": snap.retry_after_seconds,
                        "kv_cache_usage_ratio": tel.kv_cache_usage_ratio if tel else None,
                        "queue_depth": tel.queue_depth if tel else None,
                        "detail": (
                            "Backend circuit breaker is OPEN after consecutive upstream failures."
                        ),
                    },
                )
            try:
                health = await asyncio.wait_for(
                    entry.backend.health_check(timeout=timeout_seconds),
                    timeout=timeout_seconds + 0.5,
                )
            except asyncio.TimeoutError:
                return (
                    entry.route_name,
                    False,
                    {
                        "healthy": False,
                        "backend": entry.backend.name,
                        "accelerator": entry.accelerator,
                        "circuit_state": snap.state.value,
                        "detail": "Backend health probe timed out.",
                    },
                )
            except Exception as exc:
                return (
                    entry.route_name,
                    False,
                    {
                        "healthy": False,
                        "backend": entry.backend.name,
                        "accelerator": entry.accelerator,
                        "circuit_state": snap.state.value,
                        "detail": f"Backend health probe failed ({type(exc).__name__}).",
                    },
                )

            return (
                entry.route_name,
                health.healthy,
                {
                    "healthy": health.healthy,
                    "backend": health.backend,
                    "accelerator": entry.accelerator,
                    "circuit_state": snap.state.value,
                    "latency_ms": health.latency_ms,
                    "kv_cache_usage_ratio": tel.kv_cache_usage_ratio if tel else None,
                    "queue_depth": tel.queue_depth if tel else None,
                    "detail": health.detail,
                },
            )

        results = await asyncio.gather(*(_probe_one(e) for e in self._entries.values()))
        status_map: dict[str, dict[str, Any]] = {}
        any_healthy = False
        for route_name, healthy, info in results:
            status_map[route_name] = info
            if healthy:
                any_healthy = True
        return any_healthy, status_map
