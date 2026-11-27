"""Tests for SSE streaming, OpenAI API parity (/v1/chat/completions, /v1/models), and mTLS."""

from __future__ import annotations

import datetime
import json
import ssl
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from project_nexus.app import create_app
from project_nexus.backends.base import (
    BackendGenerateRequest,
    BackendGenerateResponse,
    BackendHealth,
    BackendResponseError,
    BackendStreamChunk,
    InferenceBackend,
)
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.rate_limit import ConcurrencyLimiter
from project_nexus.resilience import CircuitBreaker, CircuitState


class _MidStreamFailBackend(InferenceBackend):
    @property
    def name(self) -> str:
        return "midstream_fail"

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
        raise BackendResponseError("not used")

    async def generate_stream(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> AsyncIterator[BackendStreamChunk]:
        del request, request_id, timeout
        yield BackendStreamChunk(delta="partial ", model="test-model", backend=self.name)
        raise BackendResponseError("Upstream stream crashed mid-response.", status_code=500)

    async def health_check(self, *, timeout: float) -> BackendHealth:
        del timeout
        return BackendHealth(healthy=True, backend=self.name, detail="ok", latency_ms=0.1)


def _parse_sse_data_events(raw_text: str) -> tuple[list[dict[str, object]], bool]:
    events: list[dict[str, object]] = []
    saw_done = False
    for line in raw_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        events.append(json.loads(payload))
    return events, saw_done


def test_generate_streaming_mock_backend_and_slot_release() -> None:
    concurrency = ConcurrencyLimiter(max_concurrent=2)
    app = create_app(NexusSettings(), concurrency_limiter=concurrency, configure_loggers=False)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/generate",
            headers={"X-Request-ID": "stream-req-1"},
            json={"prompt": "Hello streaming world", "stream": True},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-request-id"] == "stream-req-1"

        events, saw_done = _parse_sse_data_events(resp.text)
        assert saw_done is True
        assert len(events) >= 2
        reconstructed = "".join(str(evt["delta"]) for evt in events)
        assert reconstructed == "[mock] Hello streaming world"
        assert events[-1]["finish_reason"] == "stop"
        assert concurrency.active == 0


def test_generate_streaming_immediate_error_returns_http_error_status() -> None:
    timeout_app = create_app(
        NexusSettings(), backend=MockBackend(mode="timeout"), configure_loggers=False
    )
    with TestClient(timeout_app) as client:
        resp = client.post("/v1/generate", json={"prompt": "hi", "stream": True})
        assert resp.status_code == 504
        assert resp.json()["error"]["code"] == "backend_timeout"

    unavail_app = create_app(
        NexusSettings(), backend=MockBackend(mode="unavailable"), configure_loggers=False
    )
    with TestClient(unavail_app) as client:
        resp2 = client.post("/v1/generate", json={"prompt": "hi", "stream": True})
        assert resp2.status_code == 503
        assert resp2.json()["error"]["code"] == "backend_unavailable"


def test_generate_streaming_midstream_failure_emits_sse_error_and_trips_breaker() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    concurrency = ConcurrencyLimiter(max_concurrent=1)
    app = create_app(
        NexusSettings(),
        backend=_MidStreamFailBackend(),
        circuit_breaker=breaker,
        concurrency_limiter=concurrency,
        configure_loggers=False,
    )
    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"prompt": "crash", "stream": True})
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "backend_error" in resp.text
        assert breaker.snapshot().state is CircuitState.OPEN
        assert concurrency.active == 0


def test_openai_chat_completions_non_streaming_and_streaming_via_mock_transport() -> None:
    captured_messages: list[object] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        captured_messages.append(body["messages"])
        if body.get("stream"):
            chunk_one = json.dumps(
                {
                    "model": "vllm-qwen",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Hello "},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            chunk_two = json.dumps(
                {
                    "model": "vllm-qwen",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "OpenAI SDK!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "total_tokens": 11,
                    },
                }
            )
            sse_body = f"data: {chunk_one}\n\ndata: {chunk_two}\n\ndata: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )
        return httpx.Response(
            200,
            json={
                "model": "vllm-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Non-streaming chat reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="vllm-qwen",
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(NexusSettings(), backend=backend, configure_loggers=False)

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hi."},
    ]
    with TestClient(app) as client:
        models_resp = client.get("/v1/models")
        assert models_resp.status_code == 200
        assert models_resp.json()["data"][0]["id"] == "vllm-qwen"

        non_stream = client.post(
            "/v1/chat/completions",
            json={"messages": messages, "max_completion_tokens": 64, "temperature": 0.1},
        )
        assert non_stream.status_code == 200
        ns_json = non_stream.json()
        assert ns_json["object"] == "chat.completion"
        assert ns_json["choices"][0]["message"]["content"] == "Non-streaming chat reply"
        assert ns_json["usage"]["total_tokens"] == 9
        assert captured_messages[0] == messages

        stream_resp = client.post(
            "/v1/chat/completions",
            json={"messages": messages, "stream": True},
        )
        assert stream_resp.status_code == 200
        assert stream_resp.headers["content-type"].startswith("text/event-stream")
        chunks, saw_done = _parse_sse_data_events(stream_resp.text)
        assert saw_done is True
        assert chunks[0]["object"] == "chat.completion.chunk"
        text = "".join(
            str(c["choices"][0]["delta"].get("content", ""))  # type: ignore[index]
            for c in chunks
        )
        assert text == "Hello OpenAI SDK!"


def test_mtls_ssl_context_and_config_pair_validation(tmp_path: Path) -> None:
    # Config must reject half-configured cert/key pairs
    with pytest.raises(ConfigError, match="client_cert_path"):
        load_settings(
            env={
                "NEXUS_BACKEND_TYPE": "openai_compatible",
                "NEXUS_BACKEND_BASE_URL": "https://vllm.internal:8443",
                "NEXUS_BACKEND_MODEL": "m",
                "NEXUS_BACKEND_CLIENT_CERT_PATH": "/tmp/client.crt",
            }
        )

    # Generate an ephemeral self-signed test cert/key pair and verify SSLContext loads it
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nexus-test-client")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "client.crt"
    key_path = tmp_path / "client.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    backend = OpenAICompatibleBackend(
        base_url="https://vllm.internal:8443",
        default_model="m",
        verify_tls=True,
        ca_bundle_path=str(cert_path),
        client_cert_path=str(cert_path),
        client_key_path=str(key_path),
    )
    verify_opt = backend._build_verify_option()
    assert isinstance(verify_opt, ssl.SSLContext)
