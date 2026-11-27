"""End-to-end API, error handling, cancellation, security, and logging tests."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from project_nexus.app import create_app
from project_nexus.backends.base import (
    BackendClientError,
    BackendGenerateRequest,
    BackendGenerateResponse,
    BackendHealth,
    InferenceBackend,
)
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.config import NexusSettings, load_settings
from project_nexus.logging_config import JsonLogFormatter
from project_nexus.metrics import NexusMetrics
from project_nexus.resilience import CircuitBreaker, CircuitState


class _CancellableBackend(InferenceBackend):
    @property
    def name(self) -> str:
        return "cancellable"

    @property
    def default_model(self) -> str:
        return "test-model"

    async def generate(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> BackendGenerateResponse:
        del request, request_id, timeout
        raise asyncio.CancelledError

    async def health_check(self, *, timeout: float) -> BackendHealth:
        del timeout
        return BackendHealth(healthy=True, backend=self.name, detail="ok", latency_ms=0.1)


class _ClientRejectingBackend(InferenceBackend):
    @property
    def name(self) -> str:
        return "rejecting"

    @property
    def default_model(self) -> str:
        return "test-model"

    async def generate(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> BackendGenerateResponse:
        del request, request_id, timeout
        raise BackendClientError("Upstream rejected unknown model.", status_code=400)

    async def health_check(self, *, timeout: float) -> BackendHealth:
        del timeout
        return BackendHealth(healthy=True, backend=self.name, detail="ok", latency_ms=0.1)


def test_healthz_readyz_and_generate_happy_path_mock_backend() -> None:
    app = create_app(NexusSettings(), configure_loggers=False)
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["backend"] == "mock"
        assert "x-request-id" in health.headers

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        gen = client.post(
            "/v1/generate",
            headers={"X-Request-ID": "req-smoke-01"},
            json={"prompt": "Hello Nexus", "max_tokens": 32, "temperature": 0.1},
        )
        assert gen.status_code == 200
        assert gen.headers["x-request-id"] == "req-smoke-01"
        body = gen.json()
        assert body["request_id"] == "req-smoke-01"
        assert body["backend"] == "mock"
        assert body["model"] == "mock-llm-v1"
        assert body["output"] == "[mock] Hello Nexus"


def test_end_to_end_forwarding_via_openai_compatible_backend() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "vllm-served-model",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Forwarded reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        return httpx.Response(404)

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="vllm-served-model",
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(NexusSettings(), backend=backend, configure_loggers=False)

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        resp = client.post("/v1/generate", json={"prompt": "Ping backend"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "openai_compatible"
        assert data["output"] == "Forwarded reply"
        assert data["usage"]["total_tokens"] == 5


def test_request_id_sanitization_prevents_log_injection() -> None:
    app = create_app(NexusSettings(), configure_loggers=False)
    with TestClient(app) as client:
        resp = client.get("/healthz", headers={"X-Request-ID": "bad\nforged-log-line"})
        assert resp.status_code == 200
        returned_id = resp.headers["x-request-id"]
        assert "\n" not in returned_id
        assert returned_id != "bad\nforged-log-line"


def test_authentication_enforcement_on_generate() -> None:
    settings = load_settings(env={"NEXUS_AUTH_API_KEY": "expected-gateway-token"})
    app = create_app(settings, configure_loggers=False)

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200

        no_auth = client.post("/v1/generate", json={"prompt": "hello"})
        assert no_auth.status_code == 401
        assert no_auth.headers["www-authenticate"] == "Bearer"

        wrong_auth = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer wrong-token"},
            json={"prompt": "hello"},
        )
        assert wrong_auth.status_code == 401

        ok_auth = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer expected-gateway-token"},
            json={"prompt": "hello"},
        )
        assert ok_auth.status_code == 200


def test_request_size_limit_content_length_and_streaming() -> None:
    settings = load_settings(env={"NEXUS_LIMITS_MAX_REQUEST_BYTES": "256"})
    app = create_app(settings, configure_loggers=False)

    with TestClient(app) as client:
        oversized = client.post("/v1/generate", json={"prompt": "x" * 400})
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "payload_too_large"

        bad_len = client.post(
            "/v1/generate",
            content=b'{"prompt":"hi"}',
            headers={"Content-Type": "application/json", "Content-Length": "abc"},
        )
        assert bad_len.status_code == 400

        # Chunked streaming request without Content-Length header must also be capped
        def chunk_gen() -> object:
            yield b'{"prompt":"'
            yield b"a" * 300
            yield b'"}'

        streamed = client.post(
            "/v1/generate",
            content=chunk_gen(),  # type: ignore[arg-type]
            headers={"Content-Type": "application/json"},
        )
        assert streamed.status_code == 413


def test_validation_errors_do_not_echo_sensitive_prompt() -> None:
    settings = load_settings(
        env={
            "NEXUS_LIMITS_MAX_PROMPT_CHARS": "20",
            "NEXUS_LIMITS_MAX_TOKENS_LIMIT": "64",
        }
    )
    app = create_app(settings, configure_loggers=False)
    secret_prompt = "CONFIDENTIAL_CUSTOMER_DATA_12345"

    with TestClient(app) as client:
        too_long = client.post("/v1/generate", json={"prompt": secret_prompt})
        assert too_long.status_code == 422
        assert secret_prompt not in too_long.text

        too_many_tokens = client.post("/v1/generate", json={"prompt": "ok", "max_tokens": 128})
        assert too_many_tokens.status_code == 422

        bad_temp = client.post(
            "/v1/generate", json={"prompt": "SUPER_SECRET_TEXT", "temperature": 9.9}
        )
        assert bad_temp.status_code == 422
        assert "SUPER_SECRET_TEXT" not in bad_temp.text


def test_backend_timeout_unavailable_error_and_circuit_breaker() -> None:
    settings = load_settings(
        env={
            "NEXUS_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "2",
            "NEXUS_CIRCUIT_BREAKER_COOLDOWN_SECONDS": "60",
        }
    )
    mock_backend = MockBackend(mode="timeout")
    app = create_app(settings, backend=mock_backend, configure_loggers=False)

    with TestClient(app) as client:
        r1 = client.post("/v1/generate", json={"prompt": "one"})
        assert r1.status_code == 504
        assert r1.json()["error"]["code"] == "backend_timeout"

        mock_backend.set_mode("unavailable")
        r2 = client.post("/v1/generate", json={"prompt": "two"})
        assert r2.status_code == 503
        assert r2.json()["error"]["code"] == "backend_unavailable"

        # Circuit breaker should now be OPEN -> fast-fail 503 without calling backend
        mock_backend.set_mode("normal")
        r3 = client.post("/v1/generate", json={"prompt": "three"})
        assert r3.status_code == 503
        assert r3.json()["error"]["code"] == "circuit_open"
        assert "retry-after" in r3.headers

        # Readiness probe also reflects OPEN circuit breaker
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["circuit_state"] == "OPEN"


def test_backend_500_maps_to_502_and_upstream_4xx_does_not_trip_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    app = create_app(
        NexusSettings(),
        backend=_ClientRejectingBackend(),
        circuit_breaker=breaker,
        configure_loggers=False,
    )
    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"prompt": "hi"})
        assert resp.status_code == 400
        assert breaker.snapshot().state is CircuitState.CLOSED

    err_app = create_app(
        NexusSettings(), backend=MockBackend(mode="error"), configure_loggers=False
    )
    with TestClient(err_app) as client:
        resp2 = client.post("/v1/generate", json={"prompt": "hi"})
        assert resp2.status_code == 502
        assert resp2.json()["error"]["code"] == "backend_error"


def test_readyz_returns_503_when_backend_unhealthy() -> None:
    backend = MockBackend(mode="unhealthy")
    app = create_app(NexusSettings(), backend=backend, configure_loggers=False)
    with TestClient(app) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"


@pytest.mark.asyncio
async def test_request_cancellation_records_metric_and_does_not_trip_circuit() -> None:
    metrics = NexusMetrics()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    app = create_app(
        NexusSettings(),
        backend=_CancellableBackend(),
        metrics=metrics,
        circuit_breaker=breaker,
        configure_loggers=False,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post("/v1/generate", json={"prompt": "cancel me"})

    assert breaker.snapshot().state is CircuitState.CLOSED
    text = metrics.render_prometheus()[0].decode("utf-8")
    assert 'outcome="cancelled"' in text


def test_structured_logs_include_request_id_and_never_log_prompts_or_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_token = "top-secret-bearer-token-xyz"
    sensitive_prompt = "PRIVATE_PROMPT_BODY_DO_NOT_LOG_999"

    settings = load_settings(env={"NEXUS_AUTH_API_KEY": secret_token})
    app = create_app(settings, configure_loggers=False)

    gateway_logger = logging.getLogger("project_nexus")
    gateway_logger.propagate = True

    with caplog.at_level(logging.INFO, logger="project_nexus"), TestClient(app) as client:
        resp = client.post(
            "/v1/generate",
            headers={
                "Authorization": f"Bearer {secret_token}",
                "X-Request-ID": "req-log-check-42",
            },
            json={"prompt": sensitive_prompt},
        )
        assert resp.status_code == 200

    formatter = JsonLogFormatter()
    formatted_lines = [
        formatter.format(rec) for rec in caplog.records if rec.name.startswith("project_nexus")
    ]
    assert formatted_lines

    parsed_logs = [json.loads(line) for line in formatted_lines]
    request_logs = [entry for entry in parsed_logs if entry.get("request_id") == "req-log-check-42"]
    assert request_logs
    assert request_logs[0]["status_code"] == 200
    assert request_logs[0]["outcome"] == "success"

    joined_logs = "\n".join(formatted_lines)
    assert sensitive_prompt not in joined_logs
    assert secret_token not in joined_logs
