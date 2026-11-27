"""Multi-backend model router with per-backend circuit breakers and optional failover."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from project_nexus.backends.base import InferenceBackend
from project_nexus.backends.factory import create_backend
from project_nexus.config import NexusSettings
from project_nexus.resilience import CircuitBreaker, CircuitState
from project_nexus.tracing import NexusTracer


@dataclass(frozen=True)
class RoutedBackendEntry:
    """Single backend pool paired with its own isolated circuit breaker."""

    route_name: str
    backend: InferenceBackend
    circuit_breaker: CircuitBreaker


class BackendRouter:
    """Routes requests to one or more configured inference backends by model name.

    Each backend pool (for example, a dedicated NVIDIA CUDA vLLM deployment and a
    separate AMD ROCm vLLM deployment) maintains an isolated ``CircuitBreaker`` so
    faults in one hardware pool do not trip unrelated pools.
    """

    def __init__(
        self,
        entries: Mapping[str, RoutedBackendEntry],
        *,
        default_route: str,
        model_routes: Mapping[str, str] | None = None,
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
        self._model_routes: dict[str, str] = dict(model_routes or {})
        self._enable_failover = enable_failover
        self._fallback_order = fallback_order
        self._multi_backend = multi_backend

    @classmethod
    def single(
        cls,
        backend: InferenceBackend,
        *,
        circuit_breaker: CircuitBreaker,
    ) -> BackendRouter:
        route_name = backend.name
        entry = RoutedBackendEntry(
            route_name=route_name,
            backend=backend,
            circuit_breaker=circuit_breaker,
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
            return cls.single(backend, circuit_breaker=breaker)

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
            )

        default_route = settings.routing.default_backend or next(iter(entries))
        return cls(
            entries,
            default_route=default_route,
            model_routes=settings.routing.model_routes,
            enable_failover=settings.routing.enable_failover,
            fallback_order=tuple(settings.routing.fallback_order),
            multi_backend=True,
        )

    @property
    def is_multi_backend(self) -> bool:
        return self._multi_backend

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

    def resolve_candidates(self, requested_model: str | None) -> list[RoutedBackendEntry]:
        """Return ordered backend candidates for a request's target model."""
        primary_route: str | None = None
        if requested_model:
            if requested_model in self._model_routes:
                primary_route = self._model_routes[requested_model]
            else:
                for name, entry in self._entries.items():
                    if entry.backend.default_model == requested_model:
                        primary_route = name
                        break

        if primary_route is None:
            primary_route = self._default_route

        ordered_names = [primary_route]
        if self._enable_failover:
            fallback_candidates = self._fallback_order or tuple(self._entries.keys())
            for candidate in fallback_candidates:
                if candidate in self._entries and candidate not in ordered_names:
                    ordered_names.append(candidate)

        return [self._entries[name] for name in ordered_names]

    def list_models(self) -> list[dict[str, str]]:
        """List all configured models across registered backend pools."""
        seen: set[str] = set()
        models: list[dict[str, str]] = []

        for model_id, pool_name in self._model_routes.items():
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append({"id": model_id, "object": "model", "owned_by": pool_name})

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

    async def check_all_health(
        self,
        *,
        timeout_seconds: float,
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Probe every registered backend pool concurrently."""

        async def _probe_one(entry: RoutedBackendEntry) -> tuple[str, bool, dict[str, Any]]:
            snap = entry.circuit_breaker.snapshot()
            if snap.state is CircuitState.OPEN:
                return (
                    entry.route_name,
                    False,
                    {
                        "healthy": False,
                        "backend": entry.backend.name,
                        "circuit_state": snap.state.value,
                        "retry_after_seconds": snap.retry_after_seconds,
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
                    "circuit_state": snap.state.value,
                    "latency_ms": health.latency_ms,
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
