"""Tests for OpenTelemetry tracing attribute safety and graceful shutdown draining."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from project_nexus.app import DrainController, create_app
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.config import NexusSettings, load_settings
from project_nexus.tracing import NexusTracer


def test_opentelemetry_spans_emit_safe_attributes_and_never_record_prompts_or_secrets() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    nexus_tracer = NexusTracer(enabled=True, tracer=provider.get_tracer("project-nexus-test"))

    secret_token = "otel-secret-token-777"
    secret_prompt = "PRIVATE_PATIENT_RECORD_DO_NOT_TRACE"
    settings = load_settings(env={"NEXUS_AUTH_API_KEY": secret_token, "NEXUS_TRACING_ENABLED": "1"})
    app = create_app(settings, tracer=nexus_tracer, configure_loggers=False)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/generate",
            headers={
                "Authorization": f"Bearer {secret_token}",
                "X-Request-ID": "trace-req-99",
            },
            json={"prompt": secret_prompt},
        )
        assert resp.status_code == 200

    spans = exporter.get_finished_spans()
    span_names = {s.name for s in spans}
    assert "nexus.http.request" in span_names
    assert "nexus.backend.generate" in span_names

    http_span = next(s for s in spans if s.name == "nexus.http.request")
    attrs = dict(http_span.attributes or {})
    assert attrs["nexus.request_id"] == "trace-req-99"
    assert attrs["http.response.status_code"] == 200
    assert attrs["nexus.outcome"] == "success"

    serialized_spans = repr([(s.name, dict(s.attributes or {})) for s in spans])
    assert secret_prompt not in serialized_spans
    assert secret_token not in serialized_spans


def test_traceparent_header_forwarded_to_upstream_backend() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    nexus_tracer = NexusTracer(enabled=True, tracer=provider.get_tracer("project-nexus-test"))

    seen_headers: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen_headers.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "traced"}}],
            },
        )

    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local:8000",
        default_model="m",
        transport=httpx.MockTransport(upstream),
        tracer=nexus_tracer,
    )
    app = create_app(NexusSettings(), backend=backend, tracer=nexus_tracer, configure_loggers=False)

    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"prompt": "trace upstream"})
        assert resp.status_code == 200

    assert "traceparent" in seen_headers


def test_readyz_returns_503_draining_while_healthz_stays_200() -> None:
    drain = DrainController()
    app = create_app(NexusSettings(), drain_controller=drain, configure_loggers=False)

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200

        drain.begin_drain()
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["status"] == "draining"

        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["draining"] is True


@pytest.mark.asyncio
async def test_drain_controller_waits_for_in_flight_requests_and_respects_timeout() -> None:
    drain = DrainController()
    drain.enter_request()

    async def release_soon() -> None:
        await asyncio.sleep(0.02)
        drain.leave_request()

    task = asyncio.create_task(release_soon())
    drained = await drain.wait_for_idle(timeout_seconds=1.0)
    await task
    assert drained is True
    assert drain.is_draining is True

    # Timeout case when a request never finishes within window
    drain2 = DrainController()
    drain2.enter_request()
    timed_out = await drain2.wait_for_idle(timeout_seconds=0.02)
    assert timed_out is False
    drain2.leave_request()
