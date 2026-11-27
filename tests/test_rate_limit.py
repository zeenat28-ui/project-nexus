"""Tests for per-caller token-bucket rate limiting and concurrency caps."""

from __future__ import annotations

from fastapi.testclient import TestClient

from project_nexus.app import create_app, create_metrics_app
from project_nexus.config import NexusSettings, load_settings
from project_nexus.metrics import NexusMetrics
from project_nexus.rate_limit import ConcurrencyLimiter, TokenBucketRateLimiter


def test_token_bucket_refill_and_bounded_lru_eviction() -> None:
    now = [1000.0]
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60,
        burst=2,
        max_tracked_callers=3,
        clock=lambda: now[0],
    )

    assert limiter.check("caller-a").allowed is True
    assert limiter.check("caller-a").allowed is True
    denied = limiter.check("caller-a")
    assert denied.allowed is False
    assert denied.retry_after_seconds >= 1

    # Advance 1.1 seconds -> 1 token refilled
    now[0] += 1.1
    assert limiter.check("caller-a").allowed is True

    # Bounded LRU prevents unbounded memory growth from rotating caller IDs
    for idx in range(20):
        limiter.check(f"rotating-caller-{idx}")
    assert limiter.tracked_callers() == 3


def test_concurrency_limiter_caps_and_releases() -> None:
    limiter = ConcurrencyLimiter(max_concurrent=2)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    assert limiter.active == 2

    limiter.release()
    assert limiter.active == 1
    assert limiter.try_acquire() is True
    limiter.release()
    limiter.release()
    assert limiter.active == 0


def test_http_rate_limiting_returns_429_and_records_metrics() -> None:
    settings = load_settings(
        env={
            "NEXUS_LIMITS_RATE_LIMIT_RPM": "60",
            "NEXUS_LIMITS_RATE_LIMIT_BURST": "2",
        }
    )
    metrics = NexusMetrics()
    now = [500.0]
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=2, clock=lambda: now[0])
    app = create_app(settings, metrics=metrics, rate_limiter=limiter, configure_loggers=False)
    metrics_app = create_metrics_app(metrics)

    with TestClient(app) as client, TestClient(metrics_app) as m_client:
        assert client.post("/v1/generate", json={"prompt": "1"}).status_code == 200
        assert client.post("/v1/generate", json={"prompt": "2"}).status_code == 200

        blocked = client.post("/v1/generate", json={"prompt": "3"})
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"
        assert "retry-after" in blocked.headers

        prom = m_client.get("/metrics").text
        assert 'nexus_rate_limited_total{endpoint="/v1/generate",reason="rate_limit"} 1.0' in prom


def test_http_concurrency_limit_returns_429_and_releases_after_request() -> None:
    metrics = NexusMetrics()
    concurrency = ConcurrencyLimiter(max_concurrent=1)
    assert concurrency.try_acquire() is True  # Simulate an in-flight call occupying the sole slot

    app = create_app(
        NexusSettings(),
        metrics=metrics,
        concurrency_limiter=concurrency,
        configure_loggers=False,
    )
    metrics_app = create_metrics_app(metrics)

    with TestClient(app) as client, TestClient(metrics_app) as m_client:
        rejected = client.post("/v1/generate", json={"prompt": "busy"})
        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == "concurrency_limited"

        concurrency.release()
        ok = client.post("/v1/generate", json={"prompt": "now free"})
        assert ok.status_code == 200
        assert concurrency.active == 0

        prom = m_client.get("/metrics").text
        assert (
            'nexus_rate_limited_total{endpoint="/v1/generate",reason="concurrency_limit"} 1.0'
            in prom
        )
