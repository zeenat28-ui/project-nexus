"""Deterministic CPU-only mock backend for local development and automated tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Literal

from project_nexus.backends.base import (
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

MockMode = Literal["normal", "timeout", "unavailable", "error", "unhealthy"]


class MockBackend(InferenceBackend):
    """Deterministic local backend requiring no GPU or external inference server."""

    def __init__(
        self,
        *,
        default_model: str = "mock-llm-v1",
        latency_ms: float = 0.0,
        mode: MockMode = "normal",
        response_prefix: str = "[mock]",
    ) -> None:
        self._default_model = default_model
        self._latency_ms = latency_ms
        self._mode: MockMode = mode
        self._response_prefix = response_prefix

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def mode(self) -> MockMode:
        return self._mode

    def set_mode(self, mode: MockMode) -> None:
        """Update simulated backend mode (primarily useful in tests)."""
        self._mode = mode

    def _render_output(self, request: BackendGenerateRequest) -> tuple[str, TokenUsage]:
        source_text = request.prompt
        if not source_text and request.messages:
            source_text = request.messages[-1].content

        compact = " ".join(source_text.split())
        snippet = compact if len(compact) <= 120 else f"{compact[:117]}..."
        output = f"{self._response_prefix} {snippet}"

        prompt_tokens = max(1, len(compact.split())) if compact else 0
        completion_tokens = min(request.max_tokens, max(1, len(output.split())))
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return output, usage

    async def _apply_delay_and_mode(self, *, timeout: float) -> None:
        if self._latency_ms > 0.0:
            delay_seconds = self._latency_ms / 1000.0
            if delay_seconds > timeout:
                await asyncio.sleep(timeout)
                raise BackendTimeoutError("Mock backend latency exceeded request timeout.")
            await asyncio.sleep(delay_seconds)

        if self._mode == "timeout":
            raise BackendTimeoutError("Mock backend simulated an upstream timeout.")
        if self._mode in {"unavailable", "unhealthy"}:
            raise BackendUnavailableError("Mock backend simulated an unavailable upstream.")
        if self._mode == "error":
            raise BackendResponseError(
                "Mock backend simulated an upstream 500 error.",
                status_code=500,
                retryable=True,
            )

    async def generate(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> BackendGenerateResponse:
        del request_id
        await self._apply_delay_and_mode(timeout=timeout)
        output, usage = self._render_output(request)
        return BackendGenerateResponse(
            output=output,
            model=request.model or self._default_model,
            backend=self.name,
            finish_reason="stop",
            usage=usage,
        )

    async def generate_stream(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> AsyncIterator[BackendStreamChunk]:
        del request_id
        await self._apply_delay_and_mode(timeout=timeout)
        output, usage = self._render_output(request)
        model = request.model or self._default_model
        words = output.split(" ")
        for idx, word in enumerate(words):
            is_last = idx == len(words) - 1
            piece = word if idx == 0 else f" {word}"
            yield BackendStreamChunk(
                delta=piece,
                model=model,
                backend=self.name,
                finish_reason="stop" if is_last else None,
                usage=usage if is_last else None,
            )

    async def health_check(self, *, timeout: float) -> BackendHealth:
        started = time.perf_counter()
        if self._latency_ms > 0.0:
            delay = min(self._latency_ms / 1000.0, timeout)
            await asyncio.sleep(delay)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if self._mode in {"unhealthy", "unavailable"}:
            return BackendHealth(
                healthy=False,
                backend=self.name,
                detail=f"Mock backend is in {self._mode!r} mode.",
                latency_ms=elapsed_ms,
            )
        if self._mode == "timeout":
            return BackendHealth(
                healthy=False,
                backend=self.name,
                detail="Mock backend health check simulated timeout.",
                latency_ms=elapsed_ms,
            )

        return BackendHealth(
            healthy=True,
            backend=self.name,
            detail="Mock backend ready (deterministic local mode).",
            latency_ms=elapsed_ms,
        )
