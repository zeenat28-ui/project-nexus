"""Optional OpenTelemetry tracing integration with strict attribute redaction.

When ``opentelemetry-api`` is installed and tracing is enabled, Nexus creates
lightweight spans for gateway and backend calls and propagates W3C traceparent
headers. Prompts, chat messages, completions, and credentials are NEVER placed
on span attributes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from project_nexus import __version__

try:  # pragma: no cover - optional extra
    from opentelemetry import propagate as _otel_propagate
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional extra
    _otel_propagate = None  # type: ignore[assignment]
    _otel_trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

_SAFE_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "nexus.request_id",
        "nexus.backend",
        "nexus.outcome",
        "nexus.stream",
        "nexus.circuit_state",
        "nexus.auth_scheme",
        "http.request.method",
        "http.response.status_code",
        "url.path",
    }
)


class _NoOpSpan:
    """Zero-overhead fallback span when tracing is disabled or OpenTelemetry is absent."""

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value


class SafeSpanHandle:
    """Wrapper ensuring only allowlisted low-sensitivity attributes are ever recorded."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def set_attribute(self, key: str, value: Any) -> None:
        if key not in _SAFE_ATTRIBUTE_KEYS or value is None:
            return
        if isinstance(value, (bool, int, float, str)):
            self._inner.set_attribute(key, value)


class NexusTracer:
    """Facade for optional OpenTelemetry tracing."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        service_name: str = "project-nexus",
        tracer: Any | None = None,
    ) -> None:
        self._service_name = service_name
        self._enabled = enabled and (tracer is not None or _OTEL_AVAILABLE)
        if tracer is not None:
            self._tracer = tracer
        elif self._enabled and _otel_trace is not None:
            self._tracer = _otel_trace.get_tracer(service_name, __version__)
        else:
            self._tracer = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._tracer is not None

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        carrier_headers: Mapping[str, str] | None = None,
    ) -> Iterator[SafeSpanHandle]:
        if not self.enabled or self._tracer is None:
            yield SafeSpanHandle(_NoOpSpan())
            return

        ctx = None
        if carrier_headers and _otel_propagate is not None:
            ctx = _otel_propagate.extract(dict(carrier_headers))

        span_cm = (
            self._tracer.start_as_current_span(name, context=ctx)
            if ctx is not None
            else self._tracer.start_as_current_span(name)
        )
        with span_cm as raw_span:
            handle = SafeSpanHandle(raw_span)
            if attributes:
                for key, val in attributes.items():
                    handle.set_attribute(key, val)
            yield handle

    def inject_headers(self, headers: dict[str, str]) -> None:
        """Inject W3C trace context into outbound HTTP headers when tracing is active."""
        if not self.enabled or _otel_propagate is None:
            return
        _otel_propagate.inject(headers)
