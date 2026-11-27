"""Tests for upstream GPU KV-cache telemetry scraping and heterogeneous CUDA+ROCm scheduling."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from project_nexus.app import create_app, create_metrics_app
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.backends.router import BackendRouter, RoutedBackendEntry
from project_nexus.config import NexusSettings
from project_nexus.metrics import NexusMetrics
from project_nexus.resilience import CircuitBreaker
from project_nexus.telemetry import EngineTelemetrySnapshot, parse_engine_prometheus_metrics


def test_parse_vllm_and_sglang_prometheus_metrics() -> None:
    vllm_sample = """
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct"} 0.72
vllm:num_requests_running{model_name="Qwen/Qwen2.5-7B-Instruct"} 4.0
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0
"""
    snap = parse_engine_prometheus_metrics(vllm_sample, now=123.0)
    assert snap.kv_cache_usage_ratio == 0.72
    assert snap.requests_running == 4
    assert snap.requests_waiting == 2
    assert snap.queue_depth == 6

    sglang_sample = """
sglang:token_usage 85.0
sglang:num_running_reqs 3
sglang:num_queue_reqs 1
"""
    sg_snap = parse_engine_prometheus_metrics(sglang_sample, now=124.0)
    assert sg_snap.kv_cache_usage_ratio == 0.85
    assert sg_snap.queue_depth == 4


@pytest.mark.asyncio
async def test_openai_backend_scrapes_upstream_vllm_metrics() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            payload = (
                "vllm:gpu_cache_usage_perc 0.81\n"
                "vllm:num_requests_running 5\n"
                "vllm:num_requests_waiting 3\n"
            )
            return httpx.Response(200, text=payload)
        return httpx.Response(200)

    backend = OpenAICompatibleBackend(
        base_url="http://vllm-rocm.local:8000",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        metrics_path="/metrics",
        transport=httpx.MockTransport(upstream),
    )
    tel = await backend.scrape_telemetry(timeout=2.0)
    await backend.shutdown()
    assert tel is not None
    assert tel.kv_cache_usage_ratio == 0.81
    assert tel.queue_depth == 8


def _make_heterogeneous_router(strategy: str) -> BackendRouter:
    rocm_entry = RoutedBackendEntry(
        route_name="rocm-pool",
        backend=MockBackend(default_model="shared-7b", response_prefix="[rocm]"),
        circuit_breaker=CircuitBreaker(),
        accelerator="amd_rocm",
        gpu_count=4,
        weight=1,
        cost_tier=1,
        max_kv_cache_usage_ratio=0.90,
    )
    cuda_entry = RoutedBackendEntry(
        route_name="cuda-pool",
        backend=MockBackend(default_model="shared-7b", response_prefix="[cuda]"),
        circuit_breaker=CircuitBreaker(),
        accelerator="nvidia_cuda",
        gpu_count=4,
        weight=3,
        cost_tier=2,
        max_kv_cache_usage_ratio=0.90,
    )
    return BackendRouter(
        {"rocm-pool": rocm_entry, "cuda-pool": cuda_entry},
        default_route="rocm-pool",
        model_routes={"shared-7b": ["rocm-pool", "cuda-pool"]},
        strategy=strategy,  # type: ignore[arg-type]
        enable_failover=True,
        multi_backend=True,
    )


def test_weighted_scheduling_across_heterogeneous_cuda_and_rocm_pools() -> None:
    router = _make_heterogeneous_router("weighted")
    picks = [router.resolve_candidates("shared-7b")[0].route_name for _ in range(8)]
    # Weight 3 (cuda-pool) : 1 (rocm-pool) -> exactly 6 cuda and 2 rocm in 8 calls
    assert picks.count("cuda-pool") == 6
    assert picks.count("rocm-pool") == 2


def test_cost_aware_and_kv_cache_saturation_spillover_and_admin_fleet() -> None:
    router = _make_heterogeneous_router("cost_aware")
    metrics = NexusMetrics()
    app = create_app(NexusSettings(), router=router, metrics=metrics, configure_loggers=False)
    admin_app = create_metrics_app(metrics, bearer_token="fleet-token", router=router)

    with TestClient(app) as client, TestClient(admin_app) as admin:
        # Initially rocm-pool (cost_tier=1) is cheaper and unsaturated -> selected first
        r1 = client.post("/v1/generate", json={"prompt": "first", "model": "shared-7b"})
        assert r1.status_code == 200
        assert r1.json()["backend"] == "rocm-pool"

        # Simulate upstream vLLM reporting 94% KV-cache utilization on rocm-pool (>= 0.90 threshold)
        router.get("rocm-pool").stats.update_telemetry(
            EngineTelemetrySnapshot(
                kv_cache_usage_ratio=0.94, requests_running=12, requests_waiting=4
            )
        )
        router.get("cuda-pool").stats.update_telemetry(
            EngineTelemetrySnapshot(
                kv_cache_usage_ratio=0.35, requests_running=2, requests_waiting=0
            )
        )

        # Gateway spills over automatically to unsaturated cuda-pool
        r2 = client.post("/v1/generate", json={"prompt": "spillover", "model": "shared-7b"})
        assert r2.status_code == 200
        assert r2.json()["backend"] == "cuda-pool"

        # Internal /admin/fleet requires auth and exposes hardware + KV-cache state
        assert admin.get("/admin/fleet").status_code == 401
        fleet = admin.get("/admin/fleet", headers={"Authorization": "Bearer fleet-token"})
        assert fleet.status_code == 200
        data = fleet.json()
        assert data["strategy"] == "cost_aware"
        assert data["pools"]["rocm-pool"]["accelerator"] == "amd_rocm"
        assert data["pools"]["rocm-pool"]["kv_saturated"] is True
        assert data["pools"]["cuda-pool"]["accelerator"] == "nvidia_cuda"
        assert data["pools"]["cuda-pool"]["kv_saturated"] is False


def test_least_inflight_and_latency_ewma_strategies() -> None:
    li_router = _make_heterogeneous_router("least_inflight")
    li_router.get("rocm-pool").stats.enter()
    li_router.get("rocm-pool").stats.enter()
    assert li_router.resolve_candidates("shared-7b")[0].route_name == "cuda-pool"

    ewma_router = _make_heterogeneous_router("latency_ewma")
    ewma_router.get("rocm-pool").stats.record_latency(450.0)
    ewma_router.get("cuda-pool").stats.record_latency(120.0)
    assert ewma_router.resolve_candidates("shared-7b")[0].route_name == "cuda-pool"
