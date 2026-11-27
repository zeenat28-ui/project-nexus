"""Repeatable backend verification harness for local and CUDA/ROCm staging pilots.

Runs health checks, non-streaming latency sweeps, and SSE streaming checks
against the configured backend and combines them with host GPU detection into a
structured JSON report without logging secrets.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from project_nexus.backends import (
    BackendError,
    BackendGenerateRequest,
    InferenceBackend,
    create_backend,
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
    """Summary percentiles for non-streaming verification requests."""

    samples: int
    min_ms: float
    p50_ms: float
    p95_ms: float
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


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1))
    return round(sorted_values[idx], 2)


async def verify_backend(
    settings: NexusSettings,
    *,
    backend: InferenceBackend | None = None,
    iterations: int = 3,
    prompt: str = "Reply with a short confirmation that inference is operational.",
    max_tokens: int = 32,
) -> VerificationReport:
    """Execute health, non-streaming, and streaming checks against a backend."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    active_backend = backend or create_backend(settings.backend)
    owns_backend = backend is None
    checks: list[CheckOutcome] = []
    latencies_ms: list[float] = []

    if owns_backend:
        await active_backend.startup()
    try:
        # 1. Health probe
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

        # 2. Non-streaming generation iterations
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

        # 3. Streaming generation check
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

    latency_summary: LatencySummary | None = None
    if latencies_ms:
        ordered = sorted(latencies_ms)
        latency_summary = LatencySummary(
            samples=len(ordered),
            min_ms=round(ordered[0], 2),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            max_ms=round(ordered[-1], 2),
        )

    gpu_report = detect_gpu_capabilities().to_dict()
    overall = all(item.passed for item in checks)
    return VerificationReport(
        overall_passed=overall,
        backend=active_backend.name,
        model=active_backend.default_model,
        checks=tuple(checks),
        latency_summary=latency_summary,
        host_gpu=gpu_report,
        note=(
            "Verification reflects only the configured backend endpoint and local host GPU "
            "visibility at execution time."
        ),
    )
