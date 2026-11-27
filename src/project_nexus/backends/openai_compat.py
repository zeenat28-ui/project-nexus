"""OpenAI-compatible HTTP inference adapter with streaming SSE and upstream mTLS.

Designed to integrate with external LLM inference servers that expose standard
OpenAI-compatible HTTP endpoints, including vLLM (CUDA and ROCm builds), SGLang,
llama.cpp server, and Text Generation Inference (TGI).
"""

from __future__ import annotations

import json
import ssl
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from project_nexus.backends.base import (
    BackendClientError,
    BackendGenerateRequest,
    BackendGenerateResponse,
    BackendHealth,
    BackendResponseError,
    BackendStreamChunk,
    BackendTimeoutError,
    BackendUnavailableError,
    InferenceBackend,
    TokenUsage,
)
from project_nexus.telemetry import EngineTelemetrySnapshot, parse_engine_prometheus_metrics
from project_nexus.tracing import NexusTracer

ApiStyle = Literal["chat_completions", "completions"]


class OpenAICompatibleBackend(InferenceBackend):
    """HTTP adapter for external OpenAI-compatible inference servers."""

    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        api_style: ApiStyle = "chat_completions",
        health_path: str = "/health",
        metrics_path: str | None = None,
        api_key: str | None = None,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 30.0,
        verify_tls: bool = True,
        ca_bundle_path: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        transport: httpx.AsyncBaseTransport | None = None,
        tracer: NexusTracer | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._api_style: ApiStyle = api_style
        self._health_path = health_path if health_path.startswith("/") else f"/{health_path}"
        self._metrics_path = (
            (metrics_path if metrics_path.startswith("/") else f"/{metrics_path}")
            if metrics_path
            else None
        )
        self._api_key = api_key
        self._connect_timeout = connect_timeout_seconds
        self._read_timeout = read_timeout_seconds
        self._verify_tls = verify_tls
        self._ca_bundle_path = ca_bundle_path
        self._client_cert_path = client_cert_path
        self._client_key_path = client_key_path
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._transport = transport
        self._tracer = tracer
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _build_verify_option(self) -> bool | ssl.SSLContext:
        if not self._verify_tls:
            if self._client_cert_path and self._client_key_path:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                ctx.load_cert_chain(
                    certfile=self._client_cert_path,
                    keyfile=self._client_key_path,
                )
                return ctx
            return False

        if self._ca_bundle_path or (self._client_cert_path and self._client_key_path):
            ctx = ssl.create_default_context(cafile=self._ca_bundle_path)
            if self._client_cert_path and self._client_key_path:
                ctx.load_cert_chain(
                    certfile=self._client_cert_path,
                    keyfile=self._client_key_path,
                )
            return ctx

        return True

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                connect=self._connect_timeout,
                read=self._read_timeout,
                write=self._connect_timeout,
                pool=self._connect_timeout,
            )
            limits = httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=self._max_keepalive_connections,
            )
            client_kwargs: dict[str, Any] = {
                "base_url": self._base_url,
                "timeout": timeout,
                "limits": limits,
                "follow_redirects": False,
                "verify": self._build_verify_option(),
            }
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**client_kwargs)
        return self._client

    async def startup(self) -> None:
        self._ensure_client()

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _build_headers(
        self, *, request_id: str | None = None, stream: bool = False
    ) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._tracer is not None:
            self._tracer.inject_headers(headers)
        return headers

    def _build_request_spec(
        self,
        request: BackendGenerateRequest,
        *,
        stream: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        model = request.model or self._default_model
        if self._api_style == "chat_completions":
            messages = (
                [{"role": m.role, "content": m.content} for m in request.messages]
                if request.messages
                else [{"role": "user", "content": request.prompt}]
            )
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": stream,
            }
            if request.stop:
                payload["stop"] = list(request.stop)
            return "/v1/chat/completions", payload

        prompt_text = request.prompt
        if not prompt_text and request.messages:
            prompt_text = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
        payload = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        return "/v1/completions", payload

    def _raise_for_status_code(self, status_code: int) -> None:
        if 200 <= status_code < 300:
            return
        if status_code in {408, 504}:
            raise BackendTimeoutError(
                f"Upstream inference backend returned timeout status HTTP {status_code}."
            )
        if status_code in {429, 502, 503}:
            raise BackendUnavailableError(
                f"Upstream inference backend is unavailable (HTTP {status_code})."
            )
        if status_code in {401, 403}:
            raise BackendResponseError(
                f"Upstream inference backend rejected gateway credentials (HTTP {status_code}).",
                status_code=status_code,
                retryable=True,
            )
        if 400 <= status_code < 500:
            raise BackendClientError(
                f"Upstream inference backend rejected request (HTTP {status_code}).",
                status_code=status_code,
            )
        raise BackendResponseError(
            f"Upstream inference backend failed with HTTP {status_code}.",
            status_code=status_code,
            retryable=True,
        )

    def _parse_response_payload(
        self,
        data: Any,
        *,
        fallback_model: str,
    ) -> BackendGenerateResponse:
        if not isinstance(data, dict):
            raise BackendResponseError(
                "Upstream inference backend returned a non-object JSON payload."
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendResponseError(
                "Upstream inference backend response did not include a valid 'choices' entry."
            )

        first_choice = choices[0]
        output_text: Any
        if self._api_style == "chat_completions":
            message = first_choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise BackendResponseError(
                    "Upstream chat completion response missing string 'choices[0].message.content'."
                )
            output_text = message["content"]
        else:
            output_text = first_choice.get("text")
            if not isinstance(output_text, str):
                raise BackendResponseError(
                    "Upstream completion response missing string 'choices[0].text'."
                )

        finish_reason_raw = first_choice.get("finish_reason")
        finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else None

        model_raw = data.get("model")
        model_name = model_raw if isinstance(model_raw, str) and model_raw else fallback_model

        usage = _parse_usage(data.get("usage"))
        return BackendGenerateResponse(
            output=output_text,
            model=model_name,
            backend=self.name,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_stream_event(
        self,
        data: Any,
        *,
        fallback_model: str,
    ) -> BackendStreamChunk | None:
        if not isinstance(data, dict):
            raise BackendResponseError("Upstream SSE event was not a JSON object.")

        model_raw = data.get("model")
        model_name = model_raw if isinstance(model_raw, str) and model_raw else fallback_model
        usage = _parse_usage(data.get("usage"))

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            if usage is not None:
                return BackendStreamChunk(
                    delta="",
                    model=model_name,
                    backend=self.name,
                    finish_reason=None,
                    usage=usage,
                )
            return None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise BackendResponseError("Upstream SSE choice entry was not an object.")

        delta_text = ""
        if self._api_style == "chat_completions":
            delta_obj = first_choice.get("delta")
            if isinstance(delta_obj, dict) and isinstance(delta_obj.get("content"), str):
                delta_text = delta_obj["content"]
        else:
            text_val = first_choice.get("text")
            if isinstance(text_val, str):
                delta_text = text_val

        finish_raw = first_choice.get("finish_reason")
        finish_reason = finish_raw if isinstance(finish_raw, str) else None

        if not delta_text and finish_reason is None and usage is None:
            return None

        return BackendStreamChunk(
            delta=delta_text,
            model=model_name,
            backend=self.name,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def generate(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> BackendGenerateResponse:
        client = self._ensure_client()
        endpoint_path, payload = self._build_request_spec(request, stream=False)
        per_request_timeout = httpx.Timeout(
            connect=min(self._connect_timeout, timeout),
            read=min(self._read_timeout, timeout),
            write=min(self._connect_timeout, timeout),
            pool=min(self._connect_timeout, timeout),
        )

        try:
            response = await client.post(
                endpoint_path,
                json=payload,
                headers=self._build_headers(request_id=request_id, stream=False),
                timeout=per_request_timeout,
            )
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("Upstream inference backend timed out.") from exc
        except (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise BackendUnavailableError(
                "Unable to connect to upstream inference backend."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(
                f"HTTP transport error talking to upstream backend ({type(exc).__name__})."
            ) from exc

        self._raise_for_status_code(response.status_code)

        try:
            data = response.json()
        except ValueError as exc:
            raise BackendResponseError("Upstream inference backend returned invalid JSON.") from exc

        return self._parse_response_payload(
            data,
            fallback_model=request.model or self._default_model,
        )

    async def generate_stream(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> AsyncIterator[BackendStreamChunk]:
        client = self._ensure_client()
        endpoint_path, payload = self._build_request_spec(request, stream=True)
        fallback_model = request.model or self._default_model
        per_request_timeout = httpx.Timeout(
            connect=min(self._connect_timeout, timeout),
            read=min(self._read_timeout, timeout),
            write=min(self._connect_timeout, timeout),
            pool=min(self._connect_timeout, timeout),
        )

        try:
            async with client.stream(
                "POST",
                endpoint_path,
                json=payload,
                headers=self._build_headers(request_id=request_id, stream=True),
                timeout=per_request_timeout,
            ) as response:
                self._raise_for_status_code(response.status_code)
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data_str)
                    except ValueError as exc:
                        raise BackendResponseError(
                            "Upstream SSE stream emitted malformed JSON."
                        ) from exc
                    chunk = self._parse_stream_event(parsed, fallback_model=fallback_model)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("Upstream streaming backend timed out.") from exc
        except (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise BackendUnavailableError(
                "Upstream streaming backend connection interrupted."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(
                f"HTTP stream transport error ({type(exc).__name__})."
            ) from exc

    async def health_check(self, *, timeout: float) -> BackendHealth:
        client = self._ensure_client()
        started = time.perf_counter()
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        probe_timeout = httpx.Timeout(
            connect=min(self._connect_timeout, timeout),
            read=timeout,
            write=min(self._connect_timeout, timeout),
            pool=min(self._connect_timeout, timeout),
        )

        try:
            response = await client.get(
                self._health_path,
                headers=headers,
                timeout=probe_timeout,
            )
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return BackendHealth(
                healthy=False,
                backend=self.name,
                detail="Upstream backend health probe timed out.",
                latency_ms=elapsed_ms,
            )
        except httpx.HTTPError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return BackendHealth(
                healthy=False,
                backend=self.name,
                detail=f"Upstream backend unreachable ({type(exc).__name__}).",
                latency_ms=elapsed_ms,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if 200 <= response.status_code < 300:
            return BackendHealth(
                healthy=True,
                backend=self.name,
                detail=f"Upstream health endpoint returned HTTP {response.status_code}.",
                latency_ms=elapsed_ms,
            )

        return BackendHealth(
            healthy=False,
            backend=self.name,
            detail=f"Upstream health endpoint returned HTTP {response.status_code}.",
            latency_ms=elapsed_ms,
        )

    async def scrape_telemetry(self, *, timeout: float) -> EngineTelemetrySnapshot | None:
        """Scrape upstream Prometheus metrics (e.g., vLLM ``/metrics``) if configured."""
        if not self._metrics_path:
            return None
        client = self._ensure_client()
        headers = {"Accept": "text/plain"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        probe_timeout = httpx.Timeout(
            connect=min(self._connect_timeout, timeout),
            read=timeout,
            write=min(self._connect_timeout, timeout),
            pool=min(self._connect_timeout, timeout),
        )
        try:
            response = await client.get(
                self._metrics_path,
                headers=headers,
                timeout=probe_timeout,
            )
        except httpx.HTTPError as exc:
            return EngineTelemetrySnapshot(
                scraped_at=time.time(),
                error=f"Telemetry scrape failed ({type(exc).__name__}).",
            )
        if not (200 <= response.status_code < 300):
            return EngineTelemetrySnapshot(
                scraped_at=time.time(),
                error=f"Telemetry endpoint returned HTTP {response.status_code}.",
            )
        return parse_engine_prometheus_metrics(response.text)


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _parse_usage(usage_raw: Any) -> TokenUsage | None:
    if not isinstance(usage_raw, dict):
        return None
    return TokenUsage(
        prompt_tokens=_opt_int(usage_raw.get("prompt_tokens")),
        completion_tokens=_opt_int(usage_raw.get("completion_tokens")),
        total_tokens=_opt_int(usage_raw.get("total_tokens")),
    )
