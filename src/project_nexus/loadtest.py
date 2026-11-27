"""Live HTTP/HTTPS end-to-end gateway load-testing harness for Project Nexus."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

import httpx

from project_nexus.verify import LatencySummary, _summarize_latencies

LoadEndpoint = Literal["/v1/generate", "/v1/chat/completions"]


@dataclass(frozen=True)
class GatewayLoadTestReport:
    """Structured report returned by ``run_gateway_load_test``."""

    target_url: str
    endpoint: str
    stream: bool
    total_requests: int
    concurrency: int
    succeeded: int
    failed: int
    error_rate: float
    status_codes: dict[str, int]
    wall_time_seconds: float
    requests_per_second: float
    latency_summary: LatencySummary | None
    ttft_summary: LatencySummary | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_gateway_load_test(
    *,
    base_url: str,
    endpoint: LoadEndpoint = "/v1/generate",
    total_requests: int = 50,
    concurrency: int = 10,
    stream: bool = False,
    model: str | None = None,
    prompt: str = "Confirm gateway availability in one short sentence.",
    max_tokens: int = 32,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    ca_bundle_path: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GatewayLoadTestReport:
    """Execute bounded-concurrency HTTP/HTTPS load against a live Nexus gateway."""
    if total_requests < 1:
        raise ValueError("total_requests must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    clean_base = base_url.rstrip("/")
    resolved_key = api_key if api_key is not None else os.environ.get("NEXUS_LOADTEST_API_KEY")
    verify: bool | ssl.SSLContext = (
        ssl.create_default_context(cafile=ca_bundle_path) if ca_bundle_path else True
    )

    body: dict[str, Any]
    if endpoint == "/v1/chat/completions":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
    else:
        body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
    if model:
        body["model"] = model

    client_kwargs: dict[str, Any] = {
        "base_url": clean_base,
        "timeout": httpx.Timeout(timeout_seconds),
        "limits": httpx.Limits(
            max_connections=max(concurrency + 8, 32),
            max_keepalive_connections=max(concurrency, 16),
        ),
        "verify": verify,
        "follow_redirects": False,
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    semaphore = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    status_counts: Counter[str] = Counter()
    succeeded = 0
    failed = 0

    async with httpx.AsyncClient(**client_kwargs) as client:

        async def _send_one(idx: int) -> None:
            nonlocal succeeded, failed
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "X-Request-ID": f"nexus-load-{idx + 1}",
            }
            if resolved_key:
                headers["Authorization"] = f"Bearer {resolved_key}"

            async with semaphore:
                started = time.perf_counter()
                try:
                    if stream:
                        first_token_ms: float | None = None
                        saw_done = False
                        async with client.stream(
                            "POST", endpoint, json=body, headers=headers
                        ) as resp:
                            status_counts[str(resp.status_code)] += 1
                            if resp.status_code != 200:
                                await resp.aread()
                                failed += 1
                                return
                            async for line in resp.aiter_lines():
                                stripped = line.strip()
                                if not stripped.startswith("data:"):
                                    continue
                                payload = stripped[5:].strip()
                                if payload == "[DONE]":
                                    saw_done = True
                                    break
                                if first_token_ms is None and payload:
                                    first_token_ms = round(
                                        (time.perf_counter() - started) * 1000.0, 2
                                    )
                        if not saw_done:
                            failed += 1
                            return
                        if first_token_ms is not None:
                            ttfts_ms.append(first_token_ms)
                    else:
                        resp = await client.post(endpoint, json=body, headers=headers)
                        status_counts[str(resp.status_code)] += 1
                        if resp.status_code != 200:
                            failed += 1
                            return
                        data = json.loads(resp.content.decode("utf-8"))
                        if not isinstance(data, dict):
                            failed += 1
                            return
                except Exception as exc:
                    status_counts[f"error:{type(exc).__name__}"] += 1
                    failed += 1
                    return

                latencies_ms.append(round((time.perf_counter() - started) * 1000.0, 2))
                succeeded += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*(_send_one(i) for i in range(total_requests)))
        wall_elapsed = max(1e-6, time.perf_counter() - wall_start)

    return GatewayLoadTestReport(
        target_url=clean_base,
        endpoint=endpoint,
        stream=stream,
        total_requests=total_requests,
        concurrency=concurrency,
        succeeded=succeeded,
        failed=failed,
        error_rate=round(failed / total_requests, 4),
        status_codes=dict(sorted(status_counts.items())),
        wall_time_seconds=round(wall_elapsed, 4),
        requests_per_second=round(succeeded / wall_elapsed, 2),
        latency_summary=_summarize_latencies(latencies_ms),
        ttft_summary=_summarize_latencies(ttfts_ms) if stream else None,
    )
