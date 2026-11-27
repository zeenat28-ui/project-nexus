"""Tests for the backend staging/hardware verification harness and CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from project_nexus.backends.mock import MockBackend
from project_nexus.backends.openai_compat import OpenAICompatibleBackend
from project_nexus.cli import main
from project_nexus.config import NexusSettings
from project_nexus.verify import verify_backend

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_verify_backend_passes_against_mock_and_openai_transport() -> None:
    report = await verify_backend(NexusSettings(), backend=MockBackend(), iterations=3)
    assert report.overall_passed is True
    assert report.backend == "mock"
    assert len(report.checks) == 3
    assert all(c.passed for c in report.checks)
    assert report.latency_summary is not None
    assert report.latency_summary.samples == 3
    assert "detected_vendors" in report.host_gpu

    def fake_vllm(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            chunk = json.dumps(
                {
                    "model": "vllm-m",
                    "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}],
                }
            )
            sse = f"data: {chunk}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "vllm-m",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    openai_backend = OpenAICompatibleBackend(
        base_url="http://vllm.staging:8000",
        default_model="vllm-m",
        transport=httpx.MockTransport(fake_vllm),
    )
    await openai_backend.startup()
    try:
        openai_report = await verify_backend(NexusSettings(), backend=openai_backend, iterations=2)
    finally:
        await openai_backend.shutdown()

    assert openai_report.overall_passed is True
    assert openai_report.backend == "openai_compatible"
    assert openai_report.latency_summary is not None
    assert openai_report.latency_summary.samples == 2


@pytest.mark.asyncio
async def test_verify_backend_reports_failure_on_unhealthy_backend() -> None:
    report = await verify_backend(
        NexusSettings(), backend=MockBackend(mode="unhealthy"), iterations=2
    )
    assert report.overall_passed is False
    assert all(not c.passed for c in report.checks)


def test_cli_verify_backend_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ok_code = main(
        [
            "verify-backend",
            "--config",
            str(REPO_ROOT / "config" / "nexus.mock.yaml"),
            "--iterations",
            "2",
        ]
    )
    assert ok_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_passed"] is True
    assert payload["latency_summary"]["samples"] == 2

    monkeypatch.setenv("NEXUS_MOCK_MODE", "unavailable")
    fail_code = main(["verify-backend", "--iterations", "1"])
    assert fail_code == 1
    fail_payload = json.loads(capsys.readouterr().out)
    assert fail_payload["overall_passed"] is False
