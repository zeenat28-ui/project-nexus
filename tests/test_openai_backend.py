"""Tests for the OpenAI-compatible HTTP backend adapter using httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from project_nexus.backends.base import (
    BackendClientError,
    BackendGenerateRequest,
    BackendResponseError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from project_nexus.backends.openai_compat import OpenAICompatibleBackend


@pytest.mark.asyncio
async def test_chat_completions_forwarding_and_headers() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["request_id"] = request.headers.get("x-request-id")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from vLLM!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 4,
                    "total_tokens": 10,
                },
            },
        )

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_style="chat_completions",
        api_key="upstream-secret-token",
        transport=httpx.MockTransport(handler),
    )
    await backend.startup()
    try:
        resp = await backend.generate(
            BackendGenerateRequest(
                prompt="Say hello",
                model="Qwen/Qwen2.5-7B-Instruct",
                max_tokens=64,
                temperature=0.2,
                stop=("END",),
            ),
            request_id="req-chat-123",
            timeout=10.0,
        )
    finally:
        await backend.shutdown()

    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer upstream-secret-token"
    assert captured["request_id"] == "req-chat-123"
    assert captured["body"] == {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 64,
        "temperature": 0.2,
        "stream": False,
        "stop": ["END"],
    }
    assert resp.output == "Hello from vLLM!"
    assert resp.finish_reason == "stop"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 10


@pytest.mark.asyncio
async def test_completions_api_style() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["prompt"] == "2 + 2 ="
        return httpx.Response(
            200,
            json={
                "model": "test-base-model",
                "choices": [{"text": " 4", "finish_reason": "length"}],
            },
        )

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="test-base-model",
        api_style="completions",
        transport=httpx.MockTransport(handler),
    )
    await backend.startup()
    try:
        resp = await backend.generate(
            BackendGenerateRequest(
                prompt="2 + 2 =",
                model="test-base-model",
                max_tokens=8,
                temperature=0.0,
            ),
            request_id="req-comp-1",
            timeout=5.0,
        )
    finally:
        await backend.shutdown()

    assert resp.output == " 4"
    assert resp.finish_reason == "length"
    assert resp.usage is None


@pytest.mark.asyncio
async def test_openai_backend_error_mappings_and_no_secret_leak() -> None:
    secret_key = "super-sensitive-upstream-key"
    req = BackendGenerateRequest(prompt="hi", model="m", max_tokens=16, temperature=0.5)

    async def call_with_handler(handler: object) -> None:
        backend = OpenAICompatibleBackend(
            base_url="http://vllm.local:8000",
            default_model="m",
            api_key=secret_key,
            transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        )
        await backend.startup()
        try:
            await backend.generate(req, request_id="r", timeout=5.0)
        finally:
            await backend.shutdown()

    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(BackendTimeoutError):
        await call_with_handler(raise_timeout)

    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BackendUnavailableError):
        await call_with_handler(raise_connect)

    with pytest.raises(BackendUnavailableError):
        await call_with_handler(lambda _r: httpx.Response(503, text="overloaded"))

    with pytest.raises(BackendTimeoutError):
        await call_with_handler(lambda _r: httpx.Response(504, text="gateway timeout"))

    with pytest.raises(BackendClientError) as client_exc:
        await call_with_handler(lambda _r: httpx.Response(400, json={"error": "unknown model"}))
    assert client_exc.value.retryable is False

    with pytest.raises(BackendResponseError) as auth_exc:
        await call_with_handler(lambda _r: httpx.Response(401, text=f"bad key {secret_key}"))
    assert secret_key not in str(auth_exc.value)

    with pytest.raises(BackendResponseError):
        await call_with_handler(lambda _r: httpx.Response(500, text="boom"))

    with pytest.raises(BackendResponseError):
        await call_with_handler(lambda _r: httpx.Response(200, text="not-json"))

    with pytest.raises(BackendResponseError):
        await call_with_handler(lambda _r: httpx.Response(200, json={"choices": []}))


@pytest.mark.asyncio
async def test_openai_backend_health_check_cases() -> None:
    ok_backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="m",
        health_path="/health",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200) if r.url.path == "/health" else httpx.Response(404)
        ),
    )
    ok_health = await ok_backend.health_check(timeout=2.0)
    await ok_backend.shutdown()
    assert ok_health.healthy is True

    bad_status_backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="m",
        transport=httpx.MockTransport(lambda _r: httpx.Response(503)),
    )
    bad_health = await bad_status_backend.health_check(timeout=2.0)
    await bad_status_backend.shutdown()
    assert bad_health.healthy is False

    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    down_backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="m",
        transport=httpx.MockTransport(raise_connect),
    )
    down_health = await down_backend.health_check(timeout=2.0)
    await down_backend.shutdown()
    assert down_health.healthy is False
    assert "ConnectError" in down_health.detail
