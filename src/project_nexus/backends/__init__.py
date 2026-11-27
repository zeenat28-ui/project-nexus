"""Inference backend adapters and multi-backend router for Project Nexus."""

from __future__ import annotations

from project_nexus.backends.base import (
    BackendClientError,
    BackendError,
    BackendGenerateRequest,
    BackendGenerateResponse,
    BackendHealth,
    BackendResponseError,
    BackendStreamChunk,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatMessage,
    InferenceBackend,
    TokenUsage,
)
from project_nexus.backends.factory import create_backend
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.backends.router import BackendRouter, RoutedBackendEntry

__all__ = [
    "BackendClientError",
    "BackendError",
    "BackendGenerateRequest",
    "BackendGenerateResponse",
    "BackendHealth",
    "BackendResponseError",
    "BackendRouter",
    "BackendStreamChunk",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "ChatMessage",
    "InferenceBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "RoutedBackendEntry",
    "TokenUsage",
    "create_backend",
]
