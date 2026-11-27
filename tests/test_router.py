"""Tests for multi-backend model routing, per-pool circuit breakers, and failover."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from project_nexus.app import create_app, create_metrics_app
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.router import BackendRouter, RoutedBackendEntry
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.metrics import NexusMetrics
from project_nexus.resilience import CircuitBreaker, CircuitState

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_shipped_multi_backend_example_config_validates() -> None:
    cfg = load_settings(REPO_ROOT / "config" / "nexus.multi-backend.example.yaml", env={})
    assert set(cfg.backends.keys()) == {"cuda-pool", "rocm-pool"}
    assert cfg.routing.default_backend == "rocm-pool"
    assert cfg.routing.strategy == "cost_aware"
    assert cfg.routing.model_routes["Qwen/Qwen2.5-7B-Instruct"] == ["rocm-pool", "cuda-pool"]
    assert cfg.backends["rocm-pool"].accelerator == "amd_rocm"
    assert cfg.backends["cuda-pool"].accelerator == "nvidia_cuda"
    assert cfg.metering.enabled is True


def test_multi_backend_config_validation_errors(tmp_path: Path) -> None:
    bad_default = tmp_path / "bad_default.yaml"
    bad_default.write_text(
        """
backends:
  pool-a:
    type: mock
routing:
  default_backend: missing-pool
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="default_backend"):
        load_settings(bad_default, env={})

    bad_route = tmp_path / "bad_route.yaml"
    bad_route.write_text(
        """
backends:
  pool-a:
    type: mock
routing:
  model_routes:
    some-model: nonexistent
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown backend pool"):
        load_settings(bad_route, env={})

    bad_name = tmp_path / "bad_name.yaml"
    bad_name.write_text(
        """
backends:
  "bad pool name!":
    type: mock
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid backend pool name"):
        load_settings(bad_name, env={})

    orphan_routing = tmp_path / "orphan.yaml"
    orphan_routing.write_text(
        """
routing:
  model_routes:
    m: pool-a
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="require named pools"):
        load_settings(orphan_routing, env={})


def _make_dual_pool_router(
    *,
    cuda_mode: str = "normal",
    rocm_mode: str = "normal",
    enable_failover: bool = False,
) -> tuple[BackendRouter, MockBackend, MockBackend]:
    cuda_backend = MockBackend(
        default_model="cuda-model",
        mode=cuda_mode,  # type: ignore[arg-type]
        response_prefix="[cuda]",
    )
    rocm_backend = MockBackend(
        default_model="rocm-model",
        mode=rocm_mode,  # type: ignore[arg-type]
        response_prefix="[rocm]",
    )
    entries = {
        "cuda-pool": RoutedBackendEntry(
            route_name="cuda-pool",
            backend=cuda_backend,
            circuit_breaker=CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0),
        ),
        "rocm-pool": RoutedBackendEntry(
            route_name="rocm-pool",
            backend=rocm_backend,
            circuit_breaker=CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0),
        ),
    }
    router = BackendRouter(
        entries,
        default_route="cuda-pool",
        model_routes={"cuda-model": "cuda-pool", "rocm-model": "rocm-pool"},
        enable_failover=enable_failover,
        fallback_order=("cuda-pool", "rocm-pool"),
        multi_backend=True,
    )
    return router, cuda_backend, rocm_backend


def test_model_routing_isolated_breakers_and_multi_pool_readyz() -> None:
    router, cuda_backend, rocm_backend = _make_dual_pool_router(enable_failover=False)
    metrics = NexusMetrics()
    app = create_app(NexusSettings(), router=router, metrics=metrics, configure_loggers=False)
    metrics_app = create_metrics_app(metrics)

    with TestClient(app) as client, TestClient(metrics_app) as m_client:
        health = client.get("/healthz").json()
        assert health["backends"] == ["cuda-pool", "rocm-pool"]

        models = client.get("/v1/models").json()["data"]
        assert {m["id"] for m in models} == {"cuda-model", "rocm-model"}

        # Route explicitly to each pool
        r_cuda = client.post("/v1/generate", json={"prompt": "hi", "model": "cuda-model"})
        assert r_cuda.status_code == 200
        assert r_cuda.json()["backend"] == "cuda-pool"
        assert r_cuda.json()["output"] == "[cuda] hi"

        r_rocm = client.post("/v1/generate", json={"prompt": "hi", "model": "rocm-model"})
        assert r_rocm.status_code == 200
        assert r_rocm.json()["backend"] == "rocm-pool"
        assert r_rocm.json()["output"] == "[rocm] hi"

        # Trip cuda-pool circuit OPEN; rocm-pool must stay completely unaffected
        cuda_backend.set_mode("unavailable")
        fail_cuda = client.post("/v1/generate", json={"prompt": "fail", "model": "cuda-model"})
        assert fail_cuda.status_code == 503
        assert router.get("cuda-pool").circuit_breaker.snapshot().state is CircuitState.OPEN
        assert router.get("rocm-pool").circuit_breaker.snapshot().state is CircuitState.CLOSED

        still_ok_rocm = client.post(
            "/v1/chat/completions",
            json={"model": "rocm-model", "messages": [{"role": "user", "content": "alive"}]},
        )
        assert still_ok_rocm.status_code == 200
        assert still_ok_rocm.json()["choices"][0]["message"]["content"] == "[rocm] alive"

        # Readyz stays 200 while rocm-pool is healthy, then turns 503 if all pools fail
        ready_partial = client.get("/readyz")
        assert ready_partial.status_code == 200
        assert ready_partial.json()["backends"]["cuda-pool"]["healthy"] is False
        assert ready_partial.json()["backends"]["rocm-pool"]["healthy"] is True

        rocm_backend.set_mode("unhealthy")
        ready_down = client.get("/readyz")
        assert ready_down.status_code == 503

        prom = m_client.get("/metrics").text
        assert 'backend="cuda-pool"' in prom
        assert 'backend="rocm-pool"' in prom


def test_opt_in_failover_works_for_non_streaming_and_streaming() -> None:
    router, cuda_backend, _rocm_backend = _make_dual_pool_router(
        cuda_mode="unavailable",
        rocm_mode="normal",
        enable_failover=True,
    )
    app = create_app(NexusSettings(), router=router, configure_loggers=False)

    with TestClient(app) as client:
        # Primary cuda-pool fails -> seamlessly falls over to rocm-pool
        resp = client.post("/v1/generate", json={"prompt": "failover test"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "rocm-pool"
        assert resp.json()["output"] == "[rocm] failover test"
        assert router.get("cuda-pool").circuit_breaker.snapshot().state is CircuitState.OPEN

        # Streaming request when cuda-pool circuit is already OPEN also fails over cleanly
        cuda_backend.set_mode("normal")  # still OPEN in circuit breaker
        stream_resp = client.post(
            "/v1/generate",
            json={"prompt": "stream failover", "stream": True},
        )
        assert stream_resp.status_code == 200
        assert "[rocm]" in stream_resp.text
        assert "data: [DONE]" in stream_resp.text
