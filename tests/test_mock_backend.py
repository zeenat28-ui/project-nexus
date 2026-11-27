"""Tests for the deterministic local mock backend."""

from __future__ import annotations

import pytest

from project_nexus.backends.base import (
    BackendGenerateRequest,
    BackendResponseError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from project_nexus.backends.mock import MockBackend


@pytest.mark.asyncio
async def test_mock_backend_is_deterministic() -> None:
    backend = MockBackend(default_model="mock-v1", response_prefix="[demo]")
    req = BackendGenerateRequest(
        prompt="Explain   what   Project Nexus does.",
        model="mock-v1",
        max_tokens=64,
        temperature=0.0,
    )

    first = await backend.generate(req, request_id="req-1", timeout=5.0)
    second = await backend.generate(req, request_id="req-2", timeout=5.0)

    assert first.output == second.output
    assert first.output == "[demo] Explain what Project Nexus does."
    assert first.model == "mock-v1"
    assert first.backend == "mock"
    assert first.finish_reason == "stop"
    assert first.usage is not None
    assert first.usage.prompt_tokens == 5


@pytest.mark.asyncio
async def test_mock_backend_failure_modes() -> None:
    req = BackendGenerateRequest(prompt="hi", model="m", max_tokens=16, temperature=0.5)

    timeout_backend = MockBackend(mode="timeout")
    with pytest.raises(BackendTimeoutError):
        await timeout_backend.generate(req, request_id="r1", timeout=2.0)

    unavail_backend = MockBackend(mode="unavailable")
    with pytest.raises(BackendUnavailableError):
        await unavail_backend.generate(req, request_id="r2", timeout=2.0)

    err_backend = MockBackend(mode="error")
    with pytest.raises(BackendResponseError):
        await err_backend.generate(req, request_id="r3", timeout=2.0)


@pytest.mark.asyncio
async def test_mock_backend_latency_exceeding_timeout() -> None:
    backend = MockBackend(latency_ms=200.0)
    req = BackendGenerateRequest(prompt="slow", model="m", max_tokens=16, temperature=0.5)
    with pytest.raises(BackendTimeoutError):
        await backend.generate(req, request_id="r-slow", timeout=0.01)


@pytest.mark.asyncio
async def test_mock_backend_health_check_reflects_mode() -> None:
    backend = MockBackend(mode="normal")
    healthy = await backend.health_check(timeout=1.0)
    assert healthy.healthy is True

    backend.set_mode("unhealthy")
    unhealthy = await backend.health_check(timeout=1.0)
    assert unhealthy.healthy is False

    backend.set_mode("timeout")
    timed_out = await backend.health_check(timeout=1.0)
    assert timed_out.healthy is False
