"""Factory for constructing configured inference backends."""

from __future__ import annotations

import httpx

from project_nexus.backends.base import InferenceBackend
from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.config import BackendSettings
from project_nexus.tracing import NexusTracer


def create_backend(
    settings: BackendSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    tracer: NexusTracer | None = None,
) -> InferenceBackend:
    """Instantiate an inference backend from validated settings."""
    if settings.type == "mock":
        return MockBackend(
            default_model=settings.model or "mock-llm-v1",
            latency_ms=settings.mock.latency_ms,
            mode=settings.mock.mode,
            response_prefix=settings.mock.response_prefix,
        )

    if settings.type == "openai_compatible":
        if not settings.base_url or not settings.model:
            raise ValueError("openai_compatible backend requires non-empty base_url and model.")
        return OpenAICompatibleBackend(
            base_url=settings.base_url,
            default_model=settings.model,
            api_style=settings.api_style,
            health_path=settings.health_path,
            metrics_path=settings.metrics_path,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            verify_tls=settings.verify_tls,
            ca_bundle_path=settings.ca_bundle_path,
            client_cert_path=settings.client_cert_path,
            client_key_path=settings.client_key_path,
            max_connections=settings.max_connections,
            max_keepalive_connections=settings.max_keepalive_connections,
            transport=transport,
            tracer=tracer,
        )

    raise ValueError(f"Unsupported backend type: {settings.type!r}")
