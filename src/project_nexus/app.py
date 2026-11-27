"""FastAPI serving gateway and internal metrics application for Project Nexus."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from project_nexus import __version__
from project_nexus.auth import (
    AuthenticatedCaller,
    AuthenticationError,
    GatewayAuthenticator,
    constant_time_bearer_matches,
)
from project_nexus.backends import (
    BackendClientError,
    BackendError,
    BackendGenerateRequest,
    BackendResponseError,
    BackendStreamChunk,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatMessage,
    InferenceBackend,
    create_backend,
)
from project_nexus.config import NexusSettings
from project_nexus.logging_config import configure_logging, sanitize_request_id
from project_nexus.metrics import NexusMetrics, normalize_endpoint_label
from project_nexus.rate_limit import ConcurrencyLimiter, TokenBucketRateLimiter
from project_nexus.resilience import CircuitBreaker, CircuitState
from project_nexus.tracing import NexusTracer

logger = logging.getLogger("project_nexus.gateway")


class DrainController:
    """Tracks graceful shutdown draining so readiness probes fail before closing sockets."""

    def __init__(self) -> None:
        self._draining = False
        self._in_flight = 0
        self._idle_event: asyncio.Event | None = None

    @property
    def is_draining(self) -> bool:
        return self._draining

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def begin_drain(self) -> None:
        self._draining = True

    def enter_request(self) -> None:
        self._in_flight += 1
        if self._idle_event is not None:
            self._idle_event.clear()

    def leave_request(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1
        if self._in_flight == 0 and self._idle_event is not None:
            self._idle_event.set()

    async def wait_for_idle(self, *, timeout_seconds: float) -> bool:
        self._draining = True
        if self._in_flight == 0:
            return True
        if timeout_seconds <= 0:
            return False
        if self._idle_event is None:
            self._idle_event = asyncio.Event()
        if self._in_flight == 0:
            return True
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False


class GenerateRequest(BaseModel):
    """Validated inference request payload for ``POST /v1/generate``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop: list[str] | None = Field(default=None, max_length=8)
    stream: bool = False


class ChatMessageInput(BaseModel):
    """Single message entry for ``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatCompletionsRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, min_length=1, max_length=256)
    messages: list[ChatMessageInput] = Field(..., min_length=1, max_length=128)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop: str | list[str] | None = None
    stream: bool = False


class TokenUsageResponse(BaseModel):
    """Optional token usage counters returned by the upstream backend."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GenerateResponse(BaseModel):
    """Normalized inference response returned to ``/v1/generate`` clients."""

    request_id: str
    model: str
    backend: str
    output: str
    finish_reason: str | None = None
    usage: TokenUsageResponse | None = None
    latency_ms: float


def _error_payload(code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


def _infer_outcome_from_status(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "success"
    if status_code in {401, 403}:
        return "auth_error"
    if status_code == 404:
        return "not_found"
    if status_code == 405:
        return "method_not_allowed"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 429:
        return "rate_limited"
    if status_code in {400, 422}:
        return "validation_error"
    if status_code == 504:
        return "timeout"
    if status_code == 503:
        return "backend_unavailable"
    if status_code == 502:
        return "backend_error"
    return "internal_error"


def _get_header(scope: Scope, name: bytes) -> str | None:
    target = name.lower()
    raw_headers: list[tuple[bytes, bytes]] = list(scope.get("headers", ()))
    for key, value in raw_headers:
        if key.lower() == target:
            try:
                return str(value.decode("latin-1"))
            except UnicodeDecodeError:
                return None
    return None


def _extract_trace_headers(scope: Scope) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in scope.get("headers", ()):
        lower_key = key.lower()
        if lower_key in {b"traceparent", b"tracestate"}:
            try:
                headers[lower_key.decode("ascii")] = value.decode("latin-1")
            except UnicodeDecodeError:
                continue
    return headers


async def _send_json_response(
    send: Send,
    *,
    status_code: int,
    payload: dict[str, Any],
    request_id: str,
) -> None:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"x-request-id", request_id.encode("ascii")),
    ]
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware enforcing request body byte limits on Content-Length and streams."""

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        request_id = str(state.get("request_id") or sanitize_request_id(None))

        content_length_raw = _get_header(scope, b"content-length")
        if content_length_raw is not None:
            try:
                declared_length = int(content_length_raw.strip())
            except ValueError:
                state["outcome"] = "validation_error"
                await _send_json_response(
                    send,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    payload=_error_payload(
                        "invalid_content_length",
                        "Invalid Content-Length header.",
                        request_id,
                    ),
                    request_id=request_id,
                )
                return
            if declared_length < 0:
                state["outcome"] = "validation_error"
                await _send_json_response(
                    send,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    payload=_error_payload(
                        "invalid_content_length",
                        "Content-Length must be non-negative.",
                        request_id,
                    ),
                    request_id=request_id,
                )
                return
            if declared_length > self.max_request_bytes:
                state["outcome"] = "payload_too_large"
                await _send_json_response(
                    send,
                    status_code=413,
                    payload=_error_payload(
                        "payload_too_large",
                        f"Request body exceeds limit of {self.max_request_bytes} bytes.",
                        request_id,
                    ),
                    request_id=request_id,
                )
                return

        method = str(scope.get("method", "GET")).upper()
        if method in {"GET", "HEAD", "OPTIONS"} and content_length_raw is None:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        total_received = 0
        while True:
            message = await receive()
            msg_type = message["type"]
            if msg_type == "http.disconnect":
                return
            if msg_type != "http.request":
                continue
            body_chunk = bytes(message.get("body", b""))
            total_received += len(body_chunk)
            if total_received > self.max_request_bytes:
                state["outcome"] = "payload_too_large"
                await _send_json_response(
                    send,
                    status_code=413,
                    payload=_error_payload(
                        "payload_too_large",
                        f"Request body exceeds limit of {self.max_request_bytes} bytes.",
                        request_id,
                    ),
                    request_id=request_id,
                )
                return
            if body_chunk:
                chunks.append(body_chunk)
            if not message.get("more_body", False):
                break

        buffered_body = b"".join(chunks)
        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": buffered_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class RequestContextMiddleware:
    """Outermost ASGI middleware for request IDs, tracing, access logs, drain, and metrics."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: NexusMetrics,
        backend_name: str,
        tracer: NexusTracer,
        drain_controller: DrainController,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.backend_name = backend_name
        self.tracer = tracer
        self.drain_controller = drain_controller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        request_id = sanitize_request_id(_get_header(scope, b"x-request-id"))
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["outcome"] = None

        path = str(scope.get("path", "/"))
        method = str(scope.get("method", "GET"))
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                has_header = any(k.lower() == b"x-request-id" for k, _ in headers)
                if not has_header:
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        trace_headers = _extract_trace_headers(scope)
        self.drain_controller.enter_request()
        try:
            with (
                self.metrics.track_in_flight(path),
                self.tracer.start_span(
                    "nexus.http.request",
                    attributes={
                        "nexus.request_id": request_id,
                        "nexus.backend": self.backend_name,
                        "http.request.method": method,
                        "url.path": normalize_endpoint_label(path),
                    },
                    carrier_headers=trace_headers,
                ) as span,
            ):
                try:
                    await self.app(scope, receive, send_with_request_id)
                except asyncio.CancelledError:
                    duration_s = time.perf_counter() - started
                    span.set_attribute("nexus.outcome", "cancelled")
                    self.metrics.observe_cancelled_request(
                        endpoint=path,
                        backend=self.backend_name,
                        duration_seconds=duration_s,
                    )
                    logger.warning(
                        "Request cancelled before completion",
                        extra={
                            "request_id": request_id,
                            "method": method,
                            "path": path,
                            "latency_ms": round(duration_s * 1000.0, 2),
                            "backend": self.backend_name,
                            "outcome": "cancelled",
                        },
                    )
                    raise
                except Exception as exc:
                    duration_s = time.perf_counter() - started
                    span.set_attribute("http.response.status_code", 500)
                    span.set_attribute("nexus.outcome", "internal_error")
                    self.metrics.observe_request(
                        endpoint=path,
                        backend=self.backend_name,
                        status_code=500,
                        outcome="internal_error",
                        duration_seconds=duration_s,
                    )
                    logger.error(
                        "Unhandled gateway exception",
                        extra={
                            "request_id": request_id,
                            "method": method,
                            "path": path,
                            "status_code": 500,
                            "latency_ms": round(duration_s * 1000.0, 2),
                            "backend": self.backend_name,
                            "outcome": "internal_error",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                duration_s = time.perf_counter() - started
                outcome = state.get("outcome") or _infer_outcome_from_status(status_code)
                span.set_attribute("http.response.status_code", status_code)
                span.set_attribute("nexus.outcome", str(outcome))
                self.metrics.observe_request(
                    endpoint=path,
                    backend=self.backend_name,
                    status_code=status_code,
                    outcome=str(outcome),
                    duration_seconds=duration_s,
                )
                log_fn = logger.info if status_code < 500 else logger.warning
                log_fn(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "latency_ms": round(duration_s * 1000.0, 2),
                        "backend": self.backend_name,
                        "outcome": str(outcome),
                    },
                )
        finally:
            self.drain_controller.leave_request()


def _metrics_response(
    request: Request, metrics: NexusMetrics, bearer_token: str | None
) -> Response:
    if bearer_token is not None:
        auth_header = request.headers.get("authorization")
        if not constant_time_bearer_matches(auth_header, bearer_token):
            request.state.outcome = "auth_error"
            request_id = getattr(request.state, "request_id", sanitize_request_id(None))
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=_error_payload(
                    "unauthorized",
                    "Missing or invalid bearer token for metrics endpoint.",
                    request_id,
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

    payload, content_type = metrics.render_prometheus()
    request.state.outcome = "success"
    return Response(content=payload, media_type=content_type)


def create_metrics_app(
    metrics: NexusMetrics,
    *,
    bearer_token: str | None = None,
) -> FastAPI:
    """Create a dedicated internal ASGI app exposing only Prometheus ``/metrics``."""
    metrics_app = FastAPI(
        title="Project Nexus Metrics",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @metrics_app.get("/metrics")
    async def scrape_metrics(request: Request) -> Response:
        return _metrics_response(request, metrics, bearer_token)

    return metrics_app


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}\n\n"


def _sse_error(code: str, message: str, request_id: str) -> str:
    body = json.dumps(
        _error_payload(code, message, request_id), ensure_ascii=True, separators=(",", ":")
    )
    return f"event: error\ndata: {body}\n\n"


def _usage_to_dict(usage: TokenUsageResponse | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _backend_error_response(
    exc: Exception,
    *,
    request: Request,
    request_id: str,
    backend_name: str,
    started: float,
    is_half_open_probe: bool,
    breaker: CircuitBreaker,
    metrics: NexusMetrics,
) -> JSONResponse:
    elapsed_s = time.perf_counter() - started
    if isinstance(exc, (asyncio.TimeoutError, BackendTimeoutError)):
        new_state = breaker.record_failure()
        metrics.set_circuit_open(backend_name, new_state is CircuitState.OPEN)
        metrics.set_backend_health(backend_name, False)
        metrics.observe_backend_call(
            backend=backend_name, outcome="timeout", duration_seconds=elapsed_s
        )
        request.state.outcome = "timeout"
        msg = (
            exc.message if isinstance(exc, BackendTimeoutError) else "Inference request timed out."
        )
        logger.warning(
            "Backend inference call timed out",
            extra={
                "request_id": request_id,
                "backend": backend_name,
                "outcome": "timeout",
                "circuit_state": new_state.value,
                "latency_ms": round(elapsed_s * 1000.0, 2),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=_error_payload("backend_timeout", msg, request_id),
        )

    if isinstance(exc, BackendUnavailableError):
        new_state = breaker.record_failure()
        metrics.set_circuit_open(backend_name, new_state is CircuitState.OPEN)
        metrics.set_backend_health(backend_name, False)
        metrics.observe_backend_call(
            backend=backend_name, outcome="backend_unavailable", duration_seconds=elapsed_s
        )
        request.state.outcome = "backend_unavailable"
        logger.warning(
            "Backend unavailable during inference",
            extra={
                "request_id": request_id,
                "backend": backend_name,
                "outcome": "backend_unavailable",
                "circuit_state": new_state.value,
                "latency_ms": round(elapsed_s * 1000.0, 2),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_payload("backend_unavailable", exc.message, request_id),
        )

    if isinstance(exc, BackendClientError):
        if is_half_open_probe:
            breaker.release_probe()
        metrics.observe_backend_call(
            backend=backend_name, outcome="validation_error", duration_seconds=elapsed_s
        )
        request.state.outcome = "validation_error"
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload("upstream_request_rejected", exc.message, request_id),
        )

    if isinstance(exc, (BackendResponseError, BackendError)):
        if exc.retryable:
            new_state = breaker.record_failure()
            metrics.set_circuit_open(backend_name, new_state is CircuitState.OPEN)
            metrics.set_backend_health(backend_name, False)
        elif is_half_open_probe:
            breaker.release_probe()
        metrics.observe_backend_call(
            backend=backend_name, outcome="backend_error", duration_seconds=elapsed_s
        )
        request.state.outcome = "backend_error"
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=_error_payload("backend_error", exc.message, request_id),
        )

    raise exc


def create_app(
    settings: NexusSettings | None = None,
    *,
    backend: InferenceBackend | None = None,
    metrics: NexusMetrics | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    authenticator: GatewayAuthenticator | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    concurrency_limiter: ConcurrencyLimiter | None = None,
    tracer: NexusTracer | None = None,
    drain_controller: DrainController | None = None,
    configure_loggers: bool = True,
) -> FastAPI:
    """Build the FastAPI serving gateway application."""
    active_settings = settings or NexusSettings()
    if configure_loggers:
        configure_logging(
            level=active_settings.logging.level,
            log_format=active_settings.logging.format,
        )

    active_tracer = tracer or NexusTracer(
        enabled=active_settings.tracing.enabled,
        service_name=active_settings.tracing.service_name,
    )
    active_backend = backend or create_backend(active_settings.backend, tracer=active_tracer)
    active_metrics = metrics or NexusMetrics()
    active_metrics.initialize_backend(active_backend.name)
    active_breaker = circuit_breaker or CircuitBreaker(
        failure_threshold=active_settings.circuit_breaker.failure_threshold,
        cooldown_seconds=active_settings.circuit_breaker.cooldown_seconds,
    )
    active_auth = authenticator or GatewayAuthenticator(active_settings.auth)
    active_rate_limiter = rate_limiter or TokenBucketRateLimiter(
        requests_per_minute=active_settings.limits.rate_limit_requests_per_minute,
        burst=active_settings.limits.rate_limit_burst,
        max_tracked_callers=active_settings.limits.max_tracked_callers,
    )
    active_concurrency = concurrency_limiter or ConcurrencyLimiter(
        max_concurrent=active_settings.limits.max_concurrent_requests
    )
    active_drain = drain_controller or DrainController()

    metrics_secret = (
        active_settings.metrics.bearer_token.get_secret_value()
        if active_settings.metrics.bearer_token is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await active_backend.startup()
        if not active_auth.is_enforced:
            logger.warning(
                "Gateway authentication is not configured; protect inference endpoints behind TLS "
                "and auth middleware or configure NEXUS_AUTH_API_KEY / JWT before exposing beyond "
                "localhost.",
                extra={
                    "backend": active_backend.name,
                    "metrics_mode": active_settings.metrics.mode,
                },
            )
        logger.info(
            "Project Nexus gateway started",
            extra={
                "backend": active_backend.name,
                "host": active_settings.server.host,
                "port": active_settings.server.port,
                "metrics_mode": active_settings.metrics.mode,
            },
        )
        try:
            yield
        finally:
            active_drain.begin_drain()
            drained = await active_drain.wait_for_idle(
                timeout_seconds=active_settings.server.shutdown_drain_seconds
            )
            if not drained:
                logger.warning(
                    "Graceful shutdown drain timeout elapsed with requests still in flight",
                    extra={"backend": active_backend.name},
                )
            await active_backend.shutdown()
            logger.info(
                "Project Nexus gateway stopped",
                extra={"backend": active_backend.name},
            )

    app = FastAPI(
        title="Project Nexus LLM Serving Gateway",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.backend = active_backend
    app.state.metrics = active_metrics
    app.state.circuit_breaker = active_breaker
    app.state.authenticator = active_auth
    app.state.rate_limiter = active_rate_limiter
    app.state.concurrency_limiter = active_concurrency
    app.state.tracer = active_tracer
    app.state.drain_controller = active_drain

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_request_bytes=active_settings.limits.max_request_bytes,
    )
    app.add_middleware(
        RequestContextMiddleware,
        metrics=active_metrics,
        backend_name=active_backend.name,
        tracer=active_tracer,
        drain_controller=active_drain,
    )

    def _authenticate(request: Request, request_id: str) -> AuthenticatedCaller | JSONResponse:
        client_host = request.client.host if request.client else None
        try:
            return active_auth.authenticate(
                request.headers.get("authorization"),
                client_host=client_host,
            )
        except AuthenticationError as exc:
            request.state.outcome = "auth_error"
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=_error_payload("unauthorized", exc.message, request_id),
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _check_limits_and_acquire(
        request: Request,
        caller: AuthenticatedCaller,
        request_id: str,
        endpoint: str,
    ) -> JSONResponse | None:
        decision = active_rate_limiter.check(caller.caller_id)
        if not decision.allowed:
            active_metrics.observe_rate_limit(endpoint=endpoint, reason="rate_limit")
            request.state.outcome = "rate_limited"
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=_error_payload(
                    "rate_limited",
                    "Caller rate limit exceeded. Retry later.",
                    request_id,
                ),
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )

        if not active_concurrency.try_acquire():
            active_metrics.observe_rate_limit(endpoint=endpoint, reason="concurrency_limit")
            request.state.outcome = "concurrency_limited"
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=_error_payload(
                    "concurrency_limited",
                    "Gateway maximum concurrent inference limit reached. Retry later.",
                    request_id,
                ),
                headers={"Retry-After": "1"},
            )

        return None

    def _acquire_circuit_or_reject(
        request: Request, request_id: str
    ) -> tuple[bool, JSONResponse | None]:
        allowed, is_half_open_probe = active_breaker.try_acquire()
        if not allowed:
            snap = active_breaker.snapshot()
            active_metrics.set_circuit_open(active_backend.name, True)
            active_metrics.set_backend_health(active_backend.name, False)
            request.state.outcome = "circuit_open"
            return False, JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=_error_payload(
                    "circuit_open",
                    "Backend circuit breaker is OPEN due to recent upstream failures.",
                    request_id,
                ),
                headers={"Retry-After": str(max(1, int(snap.retry_after_seconds)))},
            )
        return is_half_open_probe, None

    async def _run_streaming(
        *,
        request: Request,
        request_id: str,
        endpoint: str,
        backend_req: BackendGenerateRequest,
        is_half_open_probe: bool,
        openai_format: bool,
    ) -> Response:
        started = time.perf_counter()
        deadline = time.monotonic() + active_settings.limits.request_timeout_seconds
        stream_iter = active_backend.generate_stream(
            backend_req,
            request_id=request_id,
            timeout=active_settings.limits.request_timeout_seconds,
        )

        # Prefetch first chunk so immediate upstream failures return a proper HTTP error status.
        try:
            first_chunk = await asyncio.wait_for(
                anext(stream_iter),
                timeout=max(0.01, deadline - time.monotonic()),
            )
        except StopAsyncIteration:
            active_concurrency.release()
            elapsed_s = time.perf_counter() - started
            new_state = active_breaker.record_success()
            active_metrics.set_circuit_open(active_backend.name, new_state is CircuitState.OPEN)
            active_metrics.set_backend_health(active_backend.name, True)
            active_metrics.observe_backend_call(
                backend=active_backend.name, outcome="success", duration_seconds=elapsed_s
            )
            request.state.outcome = "success"
            return StreamingResponse(
                iter(["data: [DONE]\n\n"]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except asyncio.CancelledError:
            active_concurrency.release()
            if is_half_open_probe:
                active_breaker.release_probe()
            elapsed_s = time.perf_counter() - started
            active_metrics.observe_backend_call(
                backend=active_backend.name, outcome="cancelled", duration_seconds=elapsed_s
            )
            raise
        except Exception as exc:
            active_concurrency.release()
            return _backend_error_response(
                exc,
                request=request,
                request_id=request_id,
                backend_name=active_backend.name,
                started=started,
                is_half_open_probe=is_half_open_probe,
                breaker=active_breaker,
                metrics=active_metrics,
            )

        created_ts = int(time.time())
        chat_id = f"chatcmpl-{request_id}"

        def format_chunk(chunk: BackendStreamChunk, *, include_role: bool) -> str:
            if openai_format:
                delta_obj: dict[str, Any] = {}
                if include_role:
                    delta_obj["role"] = "assistant"
                if chunk.delta:
                    delta_obj["content"] = chunk.delta
                payload: dict[str, Any] = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_obj,
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                if chunk.usage is not None:
                    payload["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                return _sse_data(payload)

            gen_payload: dict[str, Any] = {
                "request_id": request_id,
                "model": chunk.model,
                "backend": chunk.backend,
                "delta": chunk.delta,
                "finish_reason": chunk.finish_reason,
            }
            if chunk.usage is not None:
                gen_payload["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            return _sse_data(gen_payload)

        async def event_generator() -> AsyncIterator[str]:
            nonlocal stream_iter
            completed_cleanly = False
            cancelled = False
            try:
                with (
                    active_metrics.track_active_stream(endpoint),
                    active_tracer.start_span(
                        "nexus.backend.generate_stream",
                        attributes={
                            "nexus.request_id": request_id,
                            "nexus.backend": active_backend.name,
                            "nexus.stream": True,
                        },
                    ) as span,
                ):
                    yield format_chunk(first_chunk, include_role=True)
                    while True:
                        if await request.is_disconnected():
                            cancelled = True
                            span.set_attribute("nexus.outcome", "cancelled")
                            return
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise BackendTimeoutError("Streaming inference request timed out.")
                        try:
                            chunk = await asyncio.wait_for(anext(stream_iter), timeout=remaining)
                        except StopAsyncIteration:
                            break
                        yield format_chunk(chunk, include_role=False)

                    yield "data: [DONE]\n\n"
                    completed_cleanly = True
                    span.set_attribute("nexus.outcome", "success")
            except asyncio.CancelledError:
                cancelled = True
                raise
            except (asyncio.TimeoutError, BackendTimeoutError) as exc:
                new_state = active_breaker.record_failure()
                active_metrics.set_circuit_open(active_backend.name, new_state is CircuitState.OPEN)
                active_metrics.set_backend_health(active_backend.name, False)
                active_metrics.observe_backend_call(
                    backend=active_backend.name,
                    outcome="timeout",
                    duration_seconds=time.perf_counter() - started,
                )
                msg = (
                    exc.message
                    if isinstance(exc, BackendTimeoutError)
                    else "Streaming inference request timed out."
                )
                yield _sse_error("backend_timeout", msg, request_id)
            except BackendError as exc:
                if exc.retryable:
                    new_state = active_breaker.record_failure()
                    active_metrics.set_circuit_open(
                        active_backend.name, new_state is CircuitState.OPEN
                    )
                    active_metrics.set_backend_health(active_backend.name, False)
                elif is_half_open_probe:
                    active_breaker.release_probe()
                active_metrics.observe_backend_call(
                    backend=active_backend.name,
                    outcome="backend_error",
                    duration_seconds=time.perf_counter() - started,
                )
                yield _sse_error("backend_error", exc.message, request_id)
            finally:
                aclose = getattr(stream_iter, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()
                active_concurrency.release()
                if completed_cleanly:
                    elapsed_s = time.perf_counter() - started
                    new_state = active_breaker.record_success()
                    active_metrics.set_circuit_open(
                        active_backend.name, new_state is CircuitState.OPEN
                    )
                    active_metrics.set_backend_health(active_backend.name, True)
                    active_metrics.observe_backend_call(
                        backend=active_backend.name,
                        outcome="success",
                        duration_seconds=elapsed_s,
                    )
                elif cancelled:
                    if is_half_open_probe:
                        active_breaker.release_probe()
                    active_metrics.observe_backend_call(
                        backend=active_backend.name,
                        outcome="cancelled",
                        duration_seconds=time.perf_counter() - started,
                    )

        request.state.outcome = "success"
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request.state.outcome = "validation_error"
        request_id = getattr(request.state, "request_id", sanitize_request_id(None))
        # Omit raw user input values so prompts or secrets are never echoed into errors/logs.
        issues = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())),
                "message": str(err.get("msg", "Invalid value")),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "request_id": request_id,
                    "details": issues,
                }
            },
        )

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        request.state.outcome = "success"
        return {
            "status": "ok",
            "version": __version__,
            "backend": active_backend.name,
            "draining": active_drain.is_draining,
        }

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        if active_drain.is_draining:
            request.state.outcome = "draining"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "draining",
                    "backend": active_backend.name,
                    "detail": "Gateway is shutting down and draining in-flight requests.",
                },
            )

        snap = active_breaker.snapshot()
        active_metrics.set_circuit_open(active_backend.name, snap.state is CircuitState.OPEN)
        if snap.state is CircuitState.OPEN:
            active_metrics.set_backend_health(active_backend.name, False)
            request.state.outcome = "circuit_open"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "backend": active_backend.name,
                    "circuit_state": snap.state.value,
                    "retry_after_seconds": snap.retry_after_seconds,
                    "detail": (
                        "Backend circuit breaker is OPEN after consecutive upstream failures."
                    ),
                },
            )

        try:
            with active_tracer.start_span(
                "nexus.backend.health_check",
                attributes={"nexus.backend": active_backend.name},
            ):
                health = await asyncio.wait_for(
                    active_backend.health_check(
                        timeout=active_settings.limits.health_check_timeout_seconds
                    ),
                    timeout=active_settings.limits.health_check_timeout_seconds + 0.5,
                )
        except asyncio.TimeoutError:
            active_metrics.set_backend_health(active_backend.name, False)
            request.state.outcome = "timeout"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "backend": active_backend.name,
                    "circuit_state": snap.state.value,
                    "detail": "Backend health probe timed out.",
                },
            )
        except Exception as exc:
            active_metrics.set_backend_health(active_backend.name, False)
            request.state.outcome = "backend_unavailable"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "backend": active_backend.name,
                    "circuit_state": snap.state.value,
                    "detail": f"Backend health probe failed ({type(exc).__name__}).",
                },
            )

        active_metrics.set_backend_health(active_backend.name, health.healthy)
        if not health.healthy:
            request.state.outcome = "backend_unavailable"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "backend": health.backend,
                    "circuit_state": snap.state.value,
                    "latency_ms": health.latency_ms,
                    "detail": health.detail,
                },
            )

        request.state.outcome = "success"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ready",
                "backend": health.backend,
                "circuit_state": snap.state.value,
                "latency_ms": health.latency_ms,
                "detail": health.detail,
            },
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        request_id = str(getattr(request.state, "request_id", sanitize_request_id(None)))
        auth_result = _authenticate(request, request_id)
        if isinstance(auth_result, JSONResponse):
            return auth_result
        request.state.outcome = "success"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "object": "list",
                "data": [
                    {
                        "id": active_backend.default_model,
                        "object": "model",
                        "owned_by": active_backend.name,
                    }
                ],
            },
        )

    @app.post("/v1/generate", response_model=None)
    async def generate(request: Request, payload: GenerateRequest) -> Response | GenerateResponse:
        request_id = str(getattr(request.state, "request_id", sanitize_request_id(None)))

        auth_result = _authenticate(request, request_id)
        if isinstance(auth_result, JSONResponse):
            return auth_result

        if len(payload.prompt) > active_settings.limits.max_prompt_chars:
            request.state.outcome = "validation_error"
            max_chars = active_settings.limits.max_prompt_chars
            return JSONResponse(
                status_code=422,
                content=_error_payload(
                    "prompt_too_long",
                    f"Prompt exceeds maximum length of {max_chars} characters.",
                    request_id,
                ),
            )

        if payload.max_tokens > active_settings.limits.max_tokens_limit:
            request.state.outcome = "validation_error"
            max_toks = active_settings.limits.max_tokens_limit
            return JSONResponse(
                status_code=422,
                content=_error_payload(
                    "max_tokens_exceeded",
                    f"max_tokens exceeds configured limit of {max_toks}.",
                    request_id,
                ),
            )

        limit_err = _check_limits_and_acquire(request, auth_result, request_id, "/v1/generate")
        if limit_err is not None:
            return limit_err

        is_half_open_probe, circuit_err = _acquire_circuit_or_reject(request, request_id)
        if circuit_err is not None:
            active_concurrency.release()
            return circuit_err

        backend_req = BackendGenerateRequest(
            prompt=payload.prompt,
            model=payload.model or active_backend.default_model,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            stop=tuple(payload.stop or ()),
        )

        if payload.stream:
            return await _run_streaming(
                request=request,
                request_id=request_id,
                endpoint="/v1/generate",
                backend_req=backend_req,
                is_half_open_probe=is_half_open_probe,
                openai_format=False,
            )

        started = time.perf_counter()
        try:
            with active_tracer.start_span(
                "nexus.backend.generate",
                attributes={
                    "nexus.request_id": request_id,
                    "nexus.backend": active_backend.name,
                    "nexus.stream": False,
                    "nexus.auth_scheme": auth_result.auth_scheme,
                },
            ):
                backend_resp = await asyncio.wait_for(
                    active_backend.generate(
                        backend_req,
                        request_id=request_id,
                        timeout=active_settings.limits.request_timeout_seconds,
                    ),
                    timeout=active_settings.limits.request_timeout_seconds,
                )
        except asyncio.CancelledError:
            if is_half_open_probe:
                active_breaker.release_probe()
            elapsed_s = time.perf_counter() - started
            active_metrics.observe_backend_call(
                backend=active_backend.name, outcome="cancelled", duration_seconds=elapsed_s
            )
            raise
        except Exception as exc:
            return _backend_error_response(
                exc,
                request=request,
                request_id=request_id,
                backend_name=active_backend.name,
                started=started,
                is_half_open_probe=is_half_open_probe,
                breaker=active_breaker,
                metrics=active_metrics,
            )
        finally:
            active_concurrency.release()

        elapsed_s = time.perf_counter() - started
        new_state = active_breaker.record_success()
        active_metrics.set_circuit_open(active_backend.name, new_state is CircuitState.OPEN)
        active_metrics.set_backend_health(active_backend.name, True)
        active_metrics.observe_backend_call(
            backend=active_backend.name, outcome="success", duration_seconds=elapsed_s
        )
        request.state.outcome = "success"

        usage_resp = (
            TokenUsageResponse(
                prompt_tokens=backend_resp.usage.prompt_tokens,
                completion_tokens=backend_resp.usage.completion_tokens,
                total_tokens=backend_resp.usage.total_tokens,
            )
            if backend_resp.usage is not None
            else None
        )
        return GenerateResponse(
            request_id=request_id,
            model=backend_resp.model,
            backend=backend_resp.backend,
            output=backend_resp.output,
            finish_reason=backend_resp.finish_reason,
            usage=usage_resp,
            latency_ms=round(elapsed_s * 1000.0, 2),
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request, payload: ChatCompletionsRequest) -> Response:
        request_id = str(getattr(request.state, "request_id", sanitize_request_id(None)))

        auth_result = _authenticate(request, request_id)
        if isinstance(auth_result, JSONResponse):
            return auth_result

        total_chars = sum(len(msg.content) for msg in payload.messages)
        if total_chars > active_settings.limits.max_prompt_chars:
            request.state.outcome = "validation_error"
            max_chars = active_settings.limits.max_prompt_chars
            return JSONResponse(
                status_code=422,
                content=_error_payload(
                    "prompt_too_long",
                    f"Combined message length exceeds maximum of {max_chars} characters.",
                    request_id,
                ),
            )

        requested_tokens = (
            payload.max_completion_tokens
            if payload.max_completion_tokens is not None
            else (payload.max_tokens if payload.max_tokens is not None else 128)
        )
        if requested_tokens > active_settings.limits.max_tokens_limit:
            request.state.outcome = "validation_error"
            max_toks = active_settings.limits.max_tokens_limit
            return JSONResponse(
                status_code=422,
                content=_error_payload(
                    "max_tokens_exceeded",
                    f"max_tokens exceeds configured limit of {max_toks}.",
                    request_id,
                ),
            )

        if isinstance(payload.stop, str):
            stop_tuple: tuple[str, ...] = (payload.stop,)
        elif isinstance(payload.stop, list):
            if len(payload.stop) > 8:
                request.state.outcome = "validation_error"
                return JSONResponse(
                    status_code=422,
                    content=_error_payload(
                        "validation_error",
                        "stop sequence list may contain at most 8 entries.",
                        request_id,
                    ),
                )
            stop_tuple = tuple(payload.stop)
        else:
            stop_tuple = ()

        limit_err = _check_limits_and_acquire(
            request, auth_result, request_id, "/v1/chat/completions"
        )
        if limit_err is not None:
            return limit_err

        is_half_open_probe, circuit_err = _acquire_circuit_or_reject(request, request_id)
        if circuit_err is not None:
            active_concurrency.release()
            return circuit_err

        chat_messages = tuple(ChatMessage(role=m.role, content=m.content) for m in payload.messages)
        backend_req = BackendGenerateRequest(
            prompt=payload.messages[-1].content,
            model=payload.model or active_backend.default_model,
            max_tokens=requested_tokens,
            temperature=payload.temperature,
            stop=stop_tuple,
            messages=chat_messages,
        )

        if payload.stream:
            return await _run_streaming(
                request=request,
                request_id=request_id,
                endpoint="/v1/chat/completions",
                backend_req=backend_req,
                is_half_open_probe=is_half_open_probe,
                openai_format=True,
            )

        started = time.perf_counter()
        try:
            with active_tracer.start_span(
                "nexus.backend.generate",
                attributes={
                    "nexus.request_id": request_id,
                    "nexus.backend": active_backend.name,
                    "nexus.stream": False,
                    "nexus.auth_scheme": auth_result.auth_scheme,
                },
            ):
                backend_resp = await asyncio.wait_for(
                    active_backend.generate(
                        backend_req,
                        request_id=request_id,
                        timeout=active_settings.limits.request_timeout_seconds,
                    ),
                    timeout=active_settings.limits.request_timeout_seconds,
                )
        except asyncio.CancelledError:
            if is_half_open_probe:
                active_breaker.release_probe()
            elapsed_s = time.perf_counter() - started
            active_metrics.observe_backend_call(
                backend=active_backend.name, outcome="cancelled", duration_seconds=elapsed_s
            )
            raise
        except Exception as exc:
            return _backend_error_response(
                exc,
                request=request,
                request_id=request_id,
                backend_name=active_backend.name,
                started=started,
                is_half_open_probe=is_half_open_probe,
                breaker=active_breaker,
                metrics=active_metrics,
            )
        finally:
            active_concurrency.release()

        elapsed_s = time.perf_counter() - started
        new_state = active_breaker.record_success()
        active_metrics.set_circuit_open(active_backend.name, new_state is CircuitState.OPEN)
        active_metrics.set_backend_health(active_backend.name, True)
        active_metrics.observe_backend_call(
            backend=active_backend.name, outcome="success", duration_seconds=elapsed_s
        )
        request.state.outcome = "success"

        usage_obj = (
            TokenUsageResponse(
                prompt_tokens=backend_resp.usage.prompt_tokens,
                completion_tokens=backend_resp.usage.completion_tokens,
                total_tokens=backend_resp.usage.total_tokens,
            )
            if backend_resp.usage is not None
            else None
        )
        body: dict[str, Any] = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": backend_resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": backend_resp.output},
                    "finish_reason": backend_resp.finish_reason,
                }
            ],
        }
        if usage_obj is not None:
            body["usage"] = _usage_to_dict(usage_obj)
        return JSONResponse(status_code=status.HTTP_200_OK, content=body)

    if active_settings.metrics.mode == "main":

        @app.get("/metrics")
        async def main_metrics(request: Request) -> Response:
            return _metrics_response(request, active_metrics, metrics_secret)

    return app
