"""Backend interface, domain models, and typed error hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from project_nexus.telemetry import EngineTelemetrySnapshot


class BackendError(Exception):
    """Base error raised by inference backend adapters."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class BackendTimeoutError(BackendError):
    """Raised when an upstream backend exceeds the configured timeout."""

    def __init__(self, message: str = "Upstream inference backend timed out.") -> None:
        super().__init__(message, status_code=504, retryable=True)


class BackendUnavailableError(BackendError):
    """Raised when an upstream backend cannot be reached or is temporarily unavailable."""

    def __init__(self, message: str = "Upstream inference backend is unavailable.") -> None:
        super().__init__(message, status_code=503, retryable=True)


class BackendClientError(BackendError):
    """Raised when an upstream backend rejects a request due to client parameters (4xx)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message, status_code=status_code, retryable=False)


class BackendResponseError(BackendError):
    """Raised when an upstream backend returns an invalid payload or unexpected 5xx/auth error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, status_code=status_code, retryable=retryable)


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message in an OpenAI-style conversation payload."""

    role: str
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Optional token usage counters reported by an inference backend."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class BackendGenerateRequest:
    """Normalized inference request passed to a backend adapter."""

    prompt: str
    model: str
    max_tokens: int
    temperature: float
    stop: tuple[str, ...] = ()
    messages: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class BackendGenerateResponse:
    """Normalized inference response returned by a backend adapter."""

    output: str
    model: str
    backend: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class BackendStreamChunk:
    """Single incremental token/text delta yielded during streaming generation."""

    delta: str
    model: str
    backend: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class BackendHealth:
    """Point-in-time health check result from a backend adapter."""

    healthy: bool
    backend: str
    detail: str
    latency_ms: float


class InferenceBackend(ABC):
    """Abstract contract for inference engine adapters integrated with Project Nexus."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Low-cardinality backend identifier (used in metrics and logs)."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name configured for this backend."""

    async def startup(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Initialize backend resources (e.g., HTTP connection pools)."""

    async def shutdown(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release backend resources cleanly on gateway shutdown."""

    @abstractmethod
    async def generate(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> BackendGenerateResponse:
        """Forward a non-streaming generation request to the underlying inference server."""

    async def generate_stream(
        self,
        request: BackendGenerateRequest,
        *,
        request_id: str,
        timeout: float,
    ) -> AsyncIterator[BackendStreamChunk]:
        """Yield incremental generation chunks (defaults to wrapping ``generate``)."""
        response = await self.generate(request, request_id=request_id, timeout=timeout)
        yield BackendStreamChunk(
            delta=response.output,
            model=response.model,
            backend=response.backend,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    @abstractmethod
    async def health_check(self, *, timeout: float) -> BackendHealth:
        """Probe whether the underlying inference backend is ready to serve requests."""

    async def scrape_telemetry(self, *, timeout: float) -> EngineTelemetrySnapshot | None:
        """Optionally scrape upstream GPU KV-cache and queue telemetry."""
        del timeout
        return None
