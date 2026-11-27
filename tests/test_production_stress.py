"""Production stress, chaos failover, Redis metering, telemetry polling, and load-test suite."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from project_nexus.app import create_app, create_metrics_app
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.backends.router import BackendRouter, RoutedBackendEntry
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.loadtest import run_gateway_load_test
from project_nexus.metering import ModelPricing, TenantBudget, UsageMeter
from project_nexus.metrics import NexusMetrics
from project_nexus.rate_limit import ConcurrencyLimiter
from project_nexus.resilience import CircuitBreaker


class _InMemorySharedRedis:
    """Thread-safe shared Redis mock simulating multi-replica cluster state."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}

    def eval(self, script: str, numkeys: int, *args: Any) -> int:
        del script
        if self.fail_with is not None:
            raise self.fail_with
        assert numkeys == 2
        usage_key, tenants_key = str(args[0]), str(args[1])
        tenant_id = str(args[2])
        prompt_t = int(args[3])
        comp_t = int(args[4])
        total_t = int(args[5])
        cost = float(args[6])

        bucket = self._hashes.setdefault(usage_key, {})
        bucket["requests"] = str(int(bucket.get("requests", "0")) + 1)
        bucket["prompt_tokens"] = str(int(bucket.get("prompt_tokens", "0")) + prompt_t)
        bucket["completion_tokens"] = str(int(bucket.get("completion_tokens", "0")) + comp_t)
        bucket["total_tokens"] = str(int(bucket.get("total_tokens", "0")) + total_t)
        bucket["estimated_cost_usd"] = (
            f"{float(bucket.get('estimated_cost_usd', '0.0')) + cost:.10f}"
        )
        self._sets.setdefault(tenants_key, set()).add(tenant_id)
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        if self.fail_with is not None:
            raise self.fail_with
        return dict(self._hashes.get(key, {}))

    def smembers(self, key: str) -> set[str]:
        if self.fail_with is not None:
            raise self.fail_with
        return set(self._sets.get(key, set()))


def test_redis_distributed_metering_synchronizes_across_replicas() -> None:
    shared_redis = _InMemorySharedRedis()
    pricing = {"m1": ModelPricing(prompt_cost_per_1k_usd=1.0, completion_cost_per_1k_usd=2.0)}
    budgets = {"team-ha": TenantBudget(max_tokens_per_window=100)}

    replica_1 = UsageMeter(
        window_seconds=3600,
        model_pricing=pricing,
        tenant_budgets=budgets,
        redis_client=shared_redis,
        clock=lambda: 1_700_000_000.0,
    )
    replica_2 = UsageMeter(
        window_seconds=3600,
        model_pricing=pricing,
        tenant_budgets=budgets,
        redis_client=shared_redis,
        clock=lambda: 1_700_000_000.0,
    )
    assert replica_1.storage_backend == "redis"

    # Replica 1 records 60 tokens; Replica 2 immediately observes 60 tokens used
    replica_1.record_usage(tenant_id="team-ha", model="m1", prompt_tokens=40, completion_tokens=20)
    assert replica_2.check_quota("team-ha").allowed is True
    assert replica_2.get_tenant_summary("team-ha")["total_tokens"] == 60

    # Replica 2 records another 50 tokens -> Replica 1 immediately blocks with quota exceeded
    replica_2.record_usage(tenant_id="team-ha", model="m1", prompt_tokens=30, completion_tokens=20)
    decision = replica_1.check_quota("team-ha")
    assert decision.allowed is False
    assert decision.reason == "token_budget"
    assert replica_1.get_all_summaries()["totals"]["total_tokens"] == 110

    # Outage fail-open vs fail-closed
    outage_redis = _InMemorySharedRedis(fail_with=ConnectionError("redis down"))
    fail_open = UsageMeter(tenant_budgets=budgets, redis_client=outage_redis, redis_fail_open=True)
    fail_closed = UsageMeter(
        tenant_budgets=budgets, redis_client=outage_redis, redis_fail_open=False
    )
    assert fail_open.check_quota("team-ha").allowed is True
    assert fail_closed.check_quota("team-ha").allowed is False


def test_redis_metering_config_validation_and_secret_redaction() -> None:
    with pytest.raises(ConfigError, match="redis_url"):
        load_settings(env={"NEXUS_METERING_BACKEND": "redis"})

    with pytest.raises(ConfigError, match="sqlite_path"):
        load_settings(env={"NEXUS_METERING_BACKEND": "sqlite"})

    secret_url = "rediss://:metering-secret-999@redis.internal:6379/1"
    cfg = load_settings(
        env={
            "NEXUS_METERING_BACKEND": "redis",
            "NEXUS_METERING_REDIS_URL": secret_url,
        }
    )
    assert cfg.to_sanitized_dict()["metering"]["redis_url"] == "***REDACTED***"
    assert "metering-secret-999" not in repr(cfg)


@pytest.mark.asyncio
async def test_background_telemetry_poller_updates_kv_saturation_without_readyz() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="vllm:gpu_cache_usage_perc 0.96\nvllm:num_requests_running 9\n",
            )
        return httpx.Response(200)

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="m",
        metrics_path="/metrics",
        transport=httpx.MockTransport(upstream),
    )
    entry = RoutedBackendEntry(
        route_name="gpu-pool",
        backend=backend,
        circuit_breaker=CircuitBreaker(),
        accelerator="nvidia_cuda",
        max_kv_cache_usage_ratio=0.90,
    )
    router = BackendRouter({"gpu-pool": entry}, default_route="gpu-pool", multi_backend=True)
    metrics = NexusMetrics()
    settings = load_settings(env={"NEXUS_ROUTING_TELEMETRY_POLL_INTERVAL_SECONDS": "0.05"})
    app = create_app(settings, router=router, metrics=metrics, configure_loggers=False)

    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.12)
        assert entry.is_kv_saturated() is True
        prom = metrics.render_prometheus()[0].decode("utf-8")
        assert 'nexus_backend_kv_cache_usage_ratio{backend="gpu-pool"} 0.96' in prom
        assert 'nexus_backend_queue_depth{backend="gpu-pool"} 9.0' in prom


@pytest.mark.asyncio
async def test_concurrent_multi_tenant_chaos_failover_and_security_stress() -> None:
    rocm_backend = MockBackend(default_model="shared-7b", latency_ms=2.0, response_prefix="[rocm]")
    cuda_backend = MockBackend(default_model="shared-7b", latency_ms=2.0, response_prefix="[cuda]")
    router = BackendRouter(
        {
            "rocm-pool": RoutedBackendEntry(
                route_name="rocm-pool",
                backend=rocm_backend,
                circuit_breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0),
                accelerator="amd_rocm",
                cost_tier=1,
            ),
            "cuda-pool": RoutedBackendEntry(
                route_name="cuda-pool",
                backend=cuda_backend,
                circuit_breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0),
                accelerator="nvidia_cuda",
                cost_tier=2,
            ),
        },
        default_route="rocm-pool",
        model_routes={"shared-7b": ["rocm-pool", "cuda-pool"]},
        strategy="cost_aware",
        enable_failover=True,
        multi_backend=True,
    )

    settings = load_settings(
        env={
            "NEXUS_LIMITS_MAX_REQUEST_BYTES": "1024",
            "NEXUS_LIMITS_MAX_CONCURRENT_REQUESTS": "64",
        },
        overrides={
            "auth": {
                "mode": "api_key",
                "named_api_keys": {
                    "tenant-steady": "steady-token-xyz",
                    "tenant-stream": "stream-token-xyz",
                    "tenant-quota": "quota-token-xyz",
                },
            },
            "metering": {
                "enabled": True,
                "default_prompt_cost_per_1k_usd": 0.5,
                "default_completion_cost_per_1k_usd": 1.0,
                "tenants": {
                    "tenant-quota": {"max_tokens_per_window": 25},
                },
            },
        },
    )

    metrics = NexusMetrics()
    concurrency = ConcurrencyLimiter(max_concurrent=64)
    app = create_app(
        settings,
        router=router,
        metrics=metrics,
        concurrency_limiter=concurrency,
        configure_loggers=False,
    )
    admin_app = create_metrics_app(
        metrics,
        bearer_token="admin-stress-key",
        usage_meter=app.state.usage_meter,
        router=router,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://nexus.test"
        ) as client,
    ):

        async def steady_worker(i: int) -> int:
            if i == 8:
                # Inject mid-flight primary pool failure under live load
                rocm_backend.set_mode("unavailable")
            resp = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer steady-token-xyz", "X-Request-ID": f"steady-{i}"},
                json={"prompt": f"CONFIDENTIAL_STEADY_{i}", "model": "shared-7b"},
            )
            return resp.status_code

        async def stream_worker(i: int) -> int:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer stream-token-xyz"},
                json={
                    "model": "shared-7b",
                    "messages": [{"role": "user", "content": f"Stream chaos {i}"}],
                    "stream": True,
                },
            )
            assert "data: [DONE]" in resp.text
            return resp.status_code

        async def quota_sequence() -> list[int]:
            codes: list[int] = []
            for i in range(12):
                resp = await client.post(
                    "/v1/generate",
                    headers={"Authorization": "Bearer quota-token-xyz"},
                    json={"prompt": f"one two three four five six {i}", "model": "shared-7b"},
                )
                codes.append(resp.status_code)
            return codes

        async def attack_worker(i: int) -> tuple[int, int, int]:
            unauth = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer forged-bad-key"},
                json={"prompt": "nope"},
            )
            oversized = await client.post(
                "/v1/generate",
                headers={"Authorization": "Bearer steady-token-xyz"},
                json={"prompt": "A" * 1500},
            )
            scanner = await client.get(
                f"/wp-admin-scan-{i}",
                headers={"X-Request-ID": "bad\nforged-id"},
            )
            return unauth.status_code, oversized.status_code, scanner.status_code

        steady_results, stream_results, quota_results, attack_results = await asyncio.gather(
            asyncio.gather(*(steady_worker(i) for i in range(25))),
            asyncio.gather(*(stream_worker(i) for i in range(15))),
            quota_sequence(),
            asyncio.gather(*(attack_worker(i) for i in range(8))),
        )

    # Zero 5xx failures on steady & streaming tenants despite primary ROCm pool crash
    assert all(code == 200 for code in steady_results)
    assert all(code == 200 for code in stream_results)
    # Quota-limited tenant receives 402 once budget exhausted
    assert 402 in quota_results
    assert all(code in {200, 402} for code in quota_results)
    # Security attackers cleanly rejected
    assert all(res == (401, 413, 404) for res in attack_results)

    # Zero leaked concurrency slots, pool in-flight gauges, or drain counters
    assert concurrency.active == 0
    assert router.get("rocm-pool").stats.snapshot()[0] == 0
    assert router.get("cuda-pool").stats.snapshot()[0] == 0
    assert app.state.drain_controller.in_flight == 0

    # Verify Prometheus cardinality & zero secret/prompt leakage
    prom_text = metrics.render_prometheus()[0].decode("utf-8")
    for forbidden in (
        "CONFIDENTIAL_STEADY_",
        "steady-token-xyz",
        "forged-bad-key",
        "/wp-admin-scan-",
    ):
        assert forbidden not in prom_text

    # Verify admin usage accounting matches completed steady/stream calls
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=admin_app), base_url="http://admin.test"
    ) as admin_client:
        usage_resp = await admin_client.get(
            "/admin/usage", headers={"Authorization": "Bearer admin-stress-key"}
        )
        assert usage_resp.status_code == 200
        tenants = usage_resp.json()["tenants"]
        assert tenants["tenant-steady"]["requests"] == 25
        assert tenants["tenant-stream"]["requests"] == 15


@pytest.mark.asyncio
async def test_gateway_load_test_harness_non_streaming_and_streaming() -> None:
    app = create_app(NexusSettings(), configure_loggers=False)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        ns = await run_gateway_load_test(
            base_url="http://nexus.test",
            endpoint="/v1/generate",
            total_requests=20,
            concurrency=5,
            stream=False,
            transport=transport,
        )
        assert ns.succeeded == 20
        assert ns.failed == 0
        assert ns.status_codes == {"200": 20}
        assert ns.latency_summary is not None

        st = await run_gateway_load_test(
            base_url="http://nexus.test",
            endpoint="/v1/chat/completions",
            total_requests=12,
            concurrency=4,
            stream=True,
            transport=transport,
        )
        assert st.succeeded == 12
        assert st.failed == 0
        assert st.ttft_summary is not None
        assert st.ttft_summary.samples == 12
        assert json.loads(json.dumps(st.to_dict()))["succeeded"] == 12
