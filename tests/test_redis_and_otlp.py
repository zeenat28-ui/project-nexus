"""Tests for Redis cluster rate limiting, OTLP exporter config, and benchmark CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from project_nexus.backends.mock import MockBackend
from project_nexus.cli import main
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.rate_limit import RedisRateLimiter
from project_nexus.tracing import configure_tracer
from project_nexus.verify import benchmark_backend

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeRedisClient:
    """In-memory stand-in simulating Redis Lua eval responses."""

    def __init__(self, *, fail_with: Exception | None = None, allow_count: int = 2) -> None:
        self.fail_with = fail_with
        self.allow_count = allow_count
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def eval(self, script: str, numkeys: int, *args: Any) -> list[Any]:
        del script
        self.calls.append((numkeys, *args))
        if self.fail_with is not None:
            raise self.fail_with
        if len(self.calls) <= self.allow_count:
            remaining = max(0, self.allow_count - len(self.calls))
            return [1, 0, str(remaining)]
        return [0, 3, "0"]

    def close(self) -> None:
        self.closed = True


def test_redis_rate_limiter_allow_deny_and_fail_policies() -> None:
    fake = _FakeRedisClient(allow_count=2)
    limiter = RedisRateLimiter(
        requests_per_minute=60,
        burst=2,
        client=fake,
        clock=lambda: 1000.0,
    )
    assert limiter.enabled is True
    assert limiter.check("caller-1").allowed is True
    assert limiter.check("caller-1").allowed is True

    denied = limiter.check("caller-1")
    assert denied.allowed is False
    assert denied.retry_after_seconds == 3
    assert fake.calls[0][1] == "nexus:rl:caller-1"

    # Fail-open policy allows request on Redis transport outage
    fail_open = RedisRateLimiter(
        requests_per_minute=60,
        client=_FakeRedisClient(fail_with=ConnectionError("redis down")),
        fail_open=True,
    )
    assert fail_open.check("caller-2").allowed is True

    # Fail-closed policy rejects request on Redis transport outage
    fail_closed = RedisRateLimiter(
        requests_per_minute=60,
        client=_FakeRedisClient(fail_with=ConnectionError("redis down")),
        fail_open=False,
    )
    decision = fail_closed.check("caller-2")
    assert decision.allowed is False
    assert decision.retry_after_seconds == 1


def test_redis_rate_limit_config_validation_and_secret_redaction() -> None:
    with pytest.raises(ConfigError, match="rate_limit_redis_url"):
        load_settings(
            env={
                "NEXUS_LIMITS_RATE_LIMIT_BACKEND": "redis",
                "NEXUS_LIMITS_RATE_LIMIT_RPM": "120",
            }
        )

    with pytest.raises(ConfigError, match=r"redis:// or rediss://"):
        load_settings(
            env={
                "NEXUS_LIMITS_RATE_LIMIT_BACKEND": "redis",
                "NEXUS_LIMITS_RATE_LIMIT_RPM": "120",
                "NEXUS_LIMITS_RATE_LIMIT_REDIS_URL": "http://not-redis:6379",
            }
        )

    secret_url = "rediss://:super-secret-redis-pass@redis.internal:6379/0"
    settings = load_settings(
        env={
            "NEXUS_LIMITS_RATE_LIMIT_BACKEND": "redis",
            "NEXUS_LIMITS_RATE_LIMIT_RPM": "120",
            "NEXUS_LIMITS_RATE_LIMIT_REDIS_URL": secret_url,
            "NEXUS_LIMITS_RATE_LIMIT_REDIS_FAIL_OPEN": "false",
        }
    )
    assert settings.limits.rate_limit_redis_fail_open is False
    sanitized = settings.to_sanitized_dict()
    assert sanitized["limits"]["rate_limit_redis_url"] == "***REDACTED***"
    assert "super-secret-redis-pass" not in repr(settings)
    assert "super-secret-redis-pass" not in repr(sanitized)


def test_configure_tracer_with_otlp_endpoint_and_validation() -> None:
    with pytest.raises(ConfigError, match="otlp_endpoint"):
        load_settings(env={"NEXUS_TRACING_OTLP_ENDPOINT": "ftp://invalid"})

    tracer = configure_tracer(
        enabled=True,
        service_name="nexus-test-otlp",
        otlp_endpoint="http://127.0.0.1:4318/v1/traces",
        sample_ratio=0.5,
        otlp_timeout_seconds=1.0,
    )
    try:
        assert tracer.enabled is True
        assert tracer.owns_provider is True
        with tracer.start_span("nexus.test", attributes={"nexus.backend": "mock"}):
            pass
    finally:
        tracer.shutdown()
    assert tracer.owns_provider is False


@pytest.mark.asyncio
async def test_benchmark_backend_non_streaming_streaming_and_failures() -> None:
    ns_report = await benchmark_backend(
        NexusSettings(),
        backend=MockBackend(),
        total_requests=8,
        concurrency=3,
        stream=False,
    )
    assert ns_report.succeeded == 8
    assert ns_report.failed == 0
    assert ns_report.error_rate == 0.0
    assert ns_report.latency_summary is not None
    assert ns_report.latency_summary.samples == 8
    assert ns_report.ttft_summary is None

    stream_report = await benchmark_backend(
        NexusSettings(),
        backend=MockBackend(),
        total_requests=6,
        concurrency=2,
        stream=True,
    )
    assert stream_report.succeeded == 6
    assert stream_report.ttft_summary is not None
    assert stream_report.ttft_summary.samples == 6

    fail_report = await benchmark_backend(
        NexusSettings(),
        backend=MockBackend(mode="unavailable"),
        total_requests=4,
        concurrency=2,
        stream=False,
    )
    assert fail_report.failed == 4
    assert fail_report.error_rate == 1.0


def test_cli_benchmark_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "benchmark",
            "--config",
            str(REPO_ROOT / "config" / "nexus.mock.yaml"),
            "--requests",
            "6",
            "--concurrency",
            "2",
            "--stream",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["succeeded"] == 6
    assert payload["ttft_summary"]["samples"] == 6

    monkeypatch.setenv("NEXUS_MOCK_MODE", "unavailable")
    fail_code = main(["benchmark", "--requests", "2", "--concurrency", "1"])
    assert fail_code == 1
