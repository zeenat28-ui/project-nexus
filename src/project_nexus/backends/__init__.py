"""Inference backend adapters for Project Nexus."""

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

__all__ = [
    "BackendClientError",
    "BackendError",
    "BackendGenerateRequest",
    "BackendGenerateResponse",
    "BackendHealth",
    "BackendResponseError",
    "BackendStreamChunk",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "ChatMessage",
    "InferenceBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "TokenUsage",
    "create_backend",
]
