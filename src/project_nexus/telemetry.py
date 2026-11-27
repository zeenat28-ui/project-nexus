"""Upstream inference engine Prometheus telemetry parsing (vLLM / SGLang).

Allows Project Nexus to observe real GPU KV-cache utilization and request queue
depth reported by upstream inference runtimes (CUDA or ROCm) without attempting
to reimplement GPU memory allocators inside the gateway.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)

_KV_CACHE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "sglang:token_usage",
    }
)

_RUNNING_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "vllm:num_requests_running",
        "sglang:num_running_reqs",
    }
)

_WAITING_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "vllm:num_requests_waiting",
        "sglang:num_queue_reqs",
    }
)


@dataclass(frozen=True)
class EngineTelemetrySnapshot:
    """Point-in-time GPU KV-cache and queue telemetry scraped from an upstream engine."""

    kv_cache_usage_ratio: float | None = None
    requests_running: int | None = None
    requests_waiting: int | None = None
    scraped_at: float = 0.0
    error: str | None = None

    @property
    def queue_depth(self) -> int | None:
        if self.requests_running is None and self.requests_waiting is None:
            return None
        return (self.requests_running or 0) + (self.requests_waiting or 0)


def parse_engine_prometheus_metrics(
    text: str, *, now: float | None = None
) -> EngineTelemetrySnapshot:
    """Extract KV-cache utilization ratio and queue counters from Prometheus text format."""
    kv_values: list[float] = []
    running_total = 0
    running_found = False
    waiting_total = 0
    waiting_found = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        if name in _KV_CACHE_METRIC_NAMES:
            # Normalize percentage (0..100) to ratio (0..1) if an engine reports > 1.0
            ratio = value / 100.0 if value > 1.0 else value
            kv_values.append(max(0.0, min(1.0, ratio)))
        elif name in _RUNNING_METRIC_NAMES:
            running_total += max(0, round(value))
            running_found = True
        elif name in _WAITING_METRIC_NAMES:
            waiting_total += max(0, round(value))
            waiting_found = True

    kv_ratio = round(max(kv_values), 4) if kv_values else None
    return EngineTelemetrySnapshot(
        kv_cache_usage_ratio=kv_ratio,
        requests_running=running_total if running_found else None,
        requests_waiting=waiting_total if waiting_found else None,
        scraped_at=time.time() if now is None else now,
    )
