"""Prometheus metrics with strictly bounded label cardinality."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

ALLOWED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/healthz",
        "/readyz",
        "/v1/generate",
        "/v1/chat/completions",
        "/v1/models",
        "/v1/usage",
        "/metrics",
        "/admin/usage",
        "/admin/fleet",
        "other",
    }
)

ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {
        "success",
        "validation_error",
        "auth_error",
        "payload_too_large",
        "not_found",
        "method_not_allowed",
        "rate_limited",
        "concurrency_limited",
        "quota_exceeded",
        "draining",
        "timeout",
        "backend_unavailable",
        "backend_error",
        "circuit_open",
        "cancelled",
        "internal_error",
    }
)

_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def normalize_endpoint_label(path: str) -> str:
    """Map arbitrary URL paths to a bounded set of known endpoints."""
    return path if path in ALLOWED_ENDPOINTS else "other"


def normalize_outcome_label(outcome: str) -> str:
    """Ensure outcome labels always come from the bounded allowlist."""
    return outcome if outcome in ALLOWED_OUTCOMES else "internal_error"


def status_class_for_code(status_code: int) -> str:
    """Map an HTTP status code to a low-cardinality class label."""
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    return "5xx"


class NexusMetrics:
    """Isolated Prometheus registry and metric recorders for a Nexus gateway instance."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)

        self.requests_total = Counter(
            "nexus_requests_total",
            "Total HTTP requests handled by the Nexus gateway.",
            labelnames=("endpoint", "backend", "status_class", "outcome"),
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "nexus_request_duration_seconds",
            "End-to-end HTTP request latency in seconds.",
            labelnames=("endpoint", "backend", "outcome"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.backend_requests_total = Counter(
            "nexus_backend_requests_total",
            "Total inference calls dispatched to the configured backend.",
            labelnames=("backend", "outcome"),
            registry=self.registry,
        )
        self.backend_request_duration_seconds = Histogram(
            "nexus_backend_request_duration_seconds",
            "Upstream inference backend latency in seconds.",
            labelnames=("backend",),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.backend_healthy = Gauge(
            "nexus_backend_healthy",
            "Latest backend health status (1=healthy, 0=unhealthy).",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.backend_circuit_open = Gauge(
            "nexus_backend_circuit_open",
            "Backend circuit breaker state (1=open, 0=closed or half-open).",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.backend_in_flight = Gauge(
            "nexus_backend_in_flight_requests",
            "Active upstream inference requests per backend pool.",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.backend_kv_cache_usage_ratio = Gauge(
            "nexus_backend_kv_cache_usage_ratio",
            "Latest scraped upstream GPU KV-cache utilization ratio (0.0-1.0).",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.backend_queue_depth = Gauge(
            "nexus_backend_queue_depth",
            "Latest scraped upstream running+waiting request count.",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.in_flight_requests = Gauge(
            "nexus_in_flight_requests",
            "Number of HTTP requests currently being processed.",
            labelnames=("endpoint",),
            registry=self.registry,
        )
        self.active_streams = Gauge(
            "nexus_active_streams",
            "Number of SSE streaming responses currently open.",
            labelnames=("endpoint",),
            registry=self.registry,
        )
        self.rate_limited_total = Counter(
            "nexus_rate_limited_total",
            "Total requests rejected by rate or concurrency limits.",
            labelnames=("endpoint", "reason"),
            registry=self.registry,
        )
        self.quota_rejected_total = Counter(
            "nexus_quota_rejected_total",
            "Total requests rejected by tenant token or cost budgets.",
            labelnames=("endpoint", "reason"),
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "nexus_tokens_total",
            "Total metered tokens processed per backend pool.",
            labelnames=("backend", "token_type"),
            registry=self.registry,
        )
        self.estimated_cost_usd_total = Counter(
            "nexus_estimated_cost_usd_total",
            "Cumulative estimated inference cost in USD per backend pool.",
            labelnames=("backend",),
            registry=self.registry,
        )

    def initialize_backend(self, backend_name: str) -> None:
        """Initialize backend gauges to conservative starting values."""
        self.backend_healthy.labels(backend=backend_name).set(0.0)
        self.backend_circuit_open.labels(backend=backend_name).set(0.0)
        self.backend_in_flight.labels(backend=backend_name).set(0.0)

    def observe_request(
        self,
        *,
        endpoint: str,
        backend: str,
        status_code: int,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        norm_endpoint = normalize_endpoint_label(endpoint)
        norm_outcome = normalize_outcome_label(outcome)
        status_class = status_class_for_code(status_code)
        self.requests_total.labels(
            endpoint=norm_endpoint,
            backend=backend,
            status_class=status_class,
            outcome=norm_outcome,
        ).inc()
        self.request_duration_seconds.labels(
            endpoint=norm_endpoint,
            backend=backend,
            outcome=norm_outcome,
        ).observe(max(0.0, duration_seconds))

    def observe_cancelled_request(
        self,
        *,
        endpoint: str,
        backend: str,
        duration_seconds: float,
    ) -> None:
        norm_endpoint = normalize_endpoint_label(endpoint)
        self.requests_total.labels(
            endpoint=norm_endpoint,
            backend=backend,
            status_class="cancelled",
            outcome="cancelled",
        ).inc()
        self.request_duration_seconds.labels(
            endpoint=norm_endpoint,
            backend=backend,
            outcome="cancelled",
        ).observe(max(0.0, duration_seconds))

    def observe_backend_call(
        self,
        *,
        backend: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        norm_outcome = normalize_outcome_label(outcome)
        self.backend_requests_total.labels(backend=backend, outcome=norm_outcome).inc()
        self.backend_request_duration_seconds.labels(backend=backend).observe(
            max(0.0, duration_seconds)
        )

    def observe_rate_limit(self, *, endpoint: str, reason: str) -> None:
        norm_endpoint = normalize_endpoint_label(endpoint)
        safe_reason = reason if reason in {"rate_limit", "concurrency_limit"} else "rate_limit"
        self.rate_limited_total.labels(endpoint=norm_endpoint, reason=safe_reason).inc()

    def observe_quota_rejection(self, *, endpoint: str, reason: str) -> None:
        norm_endpoint = normalize_endpoint_label(endpoint)
        safe_reason = reason if reason in {"token_budget", "cost_budget"} else "token_budget"
        self.quota_rejected_total.labels(endpoint=norm_endpoint, reason=safe_reason).inc()

    def observe_usage(
        self,
        *,
        backend: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        if prompt_tokens > 0:
            self.tokens_total.labels(backend=backend, token_type="prompt").inc(  # noqa: S106
                prompt_tokens
            )
        if completion_tokens > 0:
            self.tokens_total.labels(backend=backend, token_type="completion").inc(  # noqa: S106
                completion_tokens
            )
        if estimated_cost_usd > 0.0:
            self.estimated_cost_usd_total.labels(backend=backend).inc(estimated_cost_usd)

    def set_backend_health(self, backend: str, healthy: bool) -> None:
        self.backend_healthy.labels(backend=backend).set(1.0 if healthy else 0.0)

    def set_circuit_open(self, backend: str, is_open: bool) -> None:
        self.backend_circuit_open.labels(backend=backend).set(1.0 if is_open else 0.0)

    def set_backend_in_flight(self, backend: str, count: int) -> None:
        self.backend_in_flight.labels(backend=backend).set(max(0, count))

    def set_backend_telemetry(
        self,
        backend: str,
        *,
        kv_cache_usage_ratio: float | None,
        queue_depth: int | None,
    ) -> None:
        if kv_cache_usage_ratio is not None:
            self.backend_kv_cache_usage_ratio.labels(backend=backend).set(kv_cache_usage_ratio)
        if queue_depth is not None:
            self.backend_queue_depth.labels(backend=backend).set(queue_depth)

    @contextmanager
    def track_in_flight(self, endpoint: str) -> Iterator[None]:
        norm_endpoint = normalize_endpoint_label(endpoint)
        gauge = self.in_flight_requests.labels(endpoint=norm_endpoint)
        gauge.inc()
        try:
            yield
        finally:
            gauge.dec()

    @contextmanager
    def track_active_stream(self, endpoint: str) -> Iterator[None]:
        norm_endpoint = normalize_endpoint_label(endpoint)
        gauge = self.active_streams.labels(endpoint=norm_endpoint)
        gauge.inc()
        try:
            yield
        finally:
            gauge.dec()

    def render_prometheus(self) -> tuple[bytes, str]:
        """Serialize metrics in standard Prometheus text exposition format."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
