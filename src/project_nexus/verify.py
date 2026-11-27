"""Repeatable backend verification and concurrent benchmark harness.

Runs health checks, non-streaming latency sweeps, SSE streaming checks, and
bounded-concurrency benchmarks against configured backends without logging
secrets.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from project_nexus.backends import (
    BackendError,
    BackendGenerateRequest,
    BackendRouter,
    InferenceBackend,
)
from project_nexus.config import NexusSettings
from project_nexus.gpu import detect_gpu_capabilities


@dataclass(frozen=True)
class CheckOutcome:
    """Result of an individual verification check."""

    name: str
    passed: bool
    detail: str
    latency_ms: float | None = None


@dataclass(frozen=True)
class LatencySummary:
    """Summary percentiles for verification or benchmark requests."""

    samples: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class VerificationReport:
    """Structured report produced by ``verify_backend``."""

    overall_passed: bool
    backend: str
    model: str
    checks: tuple[CheckOutcome, ...]
    latency_summary: LatencySummary | None
    host_gpu: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkReport:
    """Structured concurrent load benchmark report."""

    backend: str
    model: str
    stream: bool
    total_requests: int
    concurrency: int
    succeeded: int
    failed: int
    error_rate: float
    wall_time_seconds: float
    requests_per_second: float
    latency_summary: LatencySummary | None
    ttft_summary: LatencySummary | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1))
    return round(sorted_values[idx], 2)


def _summarize_latencies(values: list[float]) -> LatencySummary | None:
    if not values:
        return None
    ordered = sorted(values)
    return LatencySummary(
        samples=len(ordered),
        min_ms=round(ordered[0], 2),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
        max_ms=round(ordered[-1], 2),
    )


def _resolve_target_backend(
    settings: NexusSettings,
    *,
    backend_name: str | None = None,
) -> tuple[str, InferenceBackend]:
    router = BackendRouter.from_settings(settings)
    if backend_name is not None:
        if backend_name not in {e.route_name for e in router.entries()}:
            available = ", ".join(e.route_name for e in router.entries())
            raise ValueError(f"Unknown backend pool {backend_name!r}. Available: {available}")
        entry = router.get(backend_name)
    else:
        entry = router.default_entry
    return entry.route_name, entry.backend


async def verify_backend(
    settings: NexusSettings,
    *,
    backend: InferenceBackend | None = None,
    backend_name: str | None = None,
    iterations: int = 3,
    prompt: str = "Reply with a short confirmation that inference is operational.",
    max_tokens: int = 32,
) -> VerificationReport:
    """Execute health, non-streaming, and streaming checks against a backend."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if backend is not None:
        resolved_name = backend.name
        active_backend = backend
        owns_backend = False
    else:
        resolved_name, active_backend = _resolve_target_backend(settings, backend_name=backend_name)
        owns_backend = True

    checks: list[CheckOutcome] = []
    latencies_ms: list[float] = []

    if owns_backend:
        await active_backend.startup()
    try:
        health = await active_backend.health_check(
            timeout=settings.limits.health_check_timeout_seconds
        )
        checks.append(
            CheckOutcome(
                name="health_check",
                passed=health.healthy,
                detail=health.detail,
                latency_ms=health.latency_ms,
            )
        )

        gen_passed = True
        gen_detail = f"Completed {iterations} non-streaming request(s)."
        req = BackendGenerateRequest(
            prompt=prompt,
            model=active_backend.default_model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        for idx in range(iterations):
            started = time.perf_counter()
            try:
                resp = await active_backend.generate(
                    req,
                    request_id=f"nexus-verify-{idx + 1}",
                    timeout=settings.limits.request_timeout_seconds,
                )
            except BackendError as exc:
                gen_passed = False
                gen_detail = f"Failed on iteration {idx + 1}: {exc.message}"
                break
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            latencies_ms.append(elapsed_ms)
            if not resp.output.strip():
                gen_passed = False
                gen_detail = f"Iteration {idx + 1} returned empty output."
                break

        checks.append(
            CheckOutcome(
                name="generate_non_streaming",
                passed=gen_passed,
                detail=gen_detail,
                latency_ms=round(sum(latencies_ms) / len(latencies_ms), 2)
                if latencies_ms
                else None,
            )
        )

        stream_started = time.perf_counter()
        ttft_ms: float | None = None
        chunk_count = 0
        collected: list[str] = []
        stream_passed = True
        stream_detail = ""
        try:
            async for chunk in active_backend.generate_stream(
                req,
                request_id="nexus-verify-stream-1",
                timeout=settings.limits.request_timeout_seconds,
            ):
                if ttft_ms is None and chunk.delta:
                    ttft_ms = round((time.perf_counter() - stream_started) * 1000.0, 2)
                if chunk.delta:
                    collected.append(chunk.delta)
                chunk_count += 1
            total_stream_ms = round((time.perf_counter() - stream_started) * 1000.0, 2)
            if not "".join(collected).strip():
                stream_passed = False
                stream_detail = "Stream completed without producing text deltas."
            else:
                stream_detail = (
                    f"Received {chunk_count} chunk(s); ttft_ms={ttft_ms}, "
                    f"total_ms={total_stream_ms}."
                )
        except BackendError as exc:
            total_stream_ms = round((time.perf_counter() - stream_started) * 1000.0, 2)
            stream_passed = False
            stream_detail = f"Streaming failed: {exc.message}"

        checks.append(
            CheckOutcome(
                name="generate_streaming",
                passed=stream_passed,
                detail=stream_detail,
                latency_ms=total_stream_ms,
            )
        )
    finally:
        if owns_backend:
            await active_backend.shutdown()

    gpu_report = detect_gpu_capabilities().to_dict()
    overall = all(item.passed for item in checks)
    return VerificationReport(
        overall_passed=overall,
        backend=resolved_name,
        model=active_backend.default_model,
        checks=tuple(checks),
        latency_summary=_summarize_latencies(latencies_ms),
        host_gpu=gpu_report,
        note=(
            "Verification reflects only the configured backend endpoint and local host GPU "
            "visibility at execution time."
        ),
    )


async def benchmark_backend(
    settings: NexusSettings,
    *,
    backend: InferenceBackend | None = None,
    backend_name: str | None = None,
    total_requests: int = 10,
    concurrency: int = 2,
    stream: bool = False,
    prompt: str = "Summarize the benefits of structured observability in one sentence.",
    max_tokens: int = 48,
) -> BenchmarkReport:
    """Run a bounded-concurrency latency and throughput benchmark against a backend."""
    if total_requests < 1:
        raise ValueError("total_requests must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    if backend is not None:
        resolved_name = backend.name
        active_backend = backend
        owns_backend = False
    else:
        resolved_name, active_backend = _resolve_target_backend(settings, backend_name=backend_name)
        owns_backend = True

    req = BackendGenerateRequest(
        prompt=prompt,
        model=active_backend.default_model,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    succeeded = 0
    failed = 0

    async def _run_one(idx: int) -> None:
        nonlocal succeeded, failed
        async with semaphore:
            started = time.perf_counter()
            try:
                if stream:
                    first_token_ms: float | None = None
                    got_text = False
                    async for chunk in active_backend.generate_stream(
                        req,
                        request_id=f"nexus-bench-{idx + 1}",
                        timeout=settings.limits.request_timeout_seconds,
                    ):
                        if chunk.delta:
                            got_text = True
                            if first_token_ms is None:
                                first_token_ms = round((time.perf_counter() - started) * 1000.0, 2)
                    if not got_text:
                        failed += 1
                        return
                    if first_token_ms is not None:
                        ttfts_ms.append(first_token_ms)
                else:
                    resp = await active_backend.generate(
                        req,
                        request_id=f"nexus-bench-{idx + 1}",
                        timeout=settings.limits.request_timeout_seconds,
                    )
                    if not resp.output.strip():
                        failed += 1
                        return
            except BackendError:
                failed += 1
                return
            latencies_ms.append(round((time.perf_counter() - started) * 1000.0, 2))
            succeeded += 1

    if owns_backend:
        await active_backend.startup()
    wall_start = time.perf_counter()
    try:
        await asyncio.gather(*(_run_one(i) for i in range(total_requests)))
    finally:
        wall_elapsed = max(1e-6, time.perf_counter() - wall_start)
        if owns_backend:
            await active_backend.shutdown()

    return BenchmarkReport(
        backend=resolved_name,
        model=active_backend.default_model,
        stream=stream,
        total_requests=total_requests,
        concurrency=concurrency,
        succeeded=succeeded,
        failed=failed,
        error_rate=round(failed / total_requests, 4),
        wall_time_seconds=round(wall_elapsed, 4),
        requests_per_second=round(succeeded / wall_elapsed, 2),
        latency_summary=_summarize_latencies(latencies_ms),
        ttft_summary=_summarize_latencies(ttfts_ms) if stream else None,
    )
