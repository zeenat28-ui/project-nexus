"""Tests for Prometheus metrics security defaults and label cardinality."""

from __future__ import annotations

from fastapi.testclient import TestClient

from project_nexus.app import create_app, create_metrics_app
from project_nexus.config import NexusSettings, load_settings
from project_nexus.metrics import NexusMetrics


def test_metrics_not_exposed_on_main_app_by_default() -> None:
    app = create_app(NexusSettings(), configure_loggers=False)
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 404


def test_separate_metrics_app_exposes_expected_prometheus_series_and_auth() -> None:
    metrics = NexusMetrics()
    gateway = create_app(NexusSettings(), metrics=metrics, configure_loggers=False)
    metrics_app = create_metrics_app(metrics, bearer_token="scrape-secret")

    with TestClient(gateway) as gw_client, TestClient(metrics_app) as m_client:
        assert gw_client.get("/readyz").status_code == 200
        assert gw_client.post("/v1/generate", json={"prompt": "metrics smoke"}).status_code == 200

        # Unauthenticated scrape must be rejected
        assert m_client.get("/metrics").status_code == 401
        assert (
            m_client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )

        # Authenticated scrape returns Prometheus exposition format
        ok = m_client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"})
        assert ok.status_code == 200
        assert "text/plain" in ok.headers["content-type"]

        body = ok.text
        for metric_name in (
            "nexus_requests_total",
            "nexus_request_duration_seconds",
            "nexus_backend_requests_total",
            "nexus_backend_request_duration_seconds",
            "nexus_backend_healthy",
            "nexus_backend_circuit_open",
            "nexus_in_flight_requests",
        ):
            assert metric_name in body


def test_metrics_on_main_app_with_bearer_token() -> None:
    settings = load_settings(
        env={
            "NEXUS_METRICS_MODE": "main",
            "NEXUS_METRICS_BEARER_TOKEN": "main-metrics-token",
        }
    )
    app = create_app(settings, configure_loggers=False)

    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer main-metrics-token"})
        assert ok.status_code == 200
        assert "nexus_requests_total" in ok.text


def test_metrics_labels_remain_low_cardinality() -> None:
    metrics = NexusMetrics()
    gateway = create_app(NexusSettings(), metrics=metrics, configure_loggers=False)
    metrics_app = create_metrics_app(metrics)

    with TestClient(gateway) as gw_client, TestClient(metrics_app) as m_client:
        # Send requests with distinct prompts, request IDs, models, and arbitrary 404 paths
        for idx in range(5):
            gw_client.post(
                "/v1/generate",
                headers={"X-Request-ID": f"unique-req-{idx}"},
                json={
                    "prompt": f"unique-prompt-{idx}",
                    "model": f"user-supplied-model-{idx}",
                },
            )
            gw_client.get(f"/unlisted-random-path-{idx}")

        body = m_client.get("/metrics").text

    # No high-cardinality values should ever leak into Prometheus labels
    for forbidden in (
        "unique-req-",
        "unique-prompt-",
        "user-supplied-model-",
        "/unlisted-random-path-",
    ):
        assert forbidden not in body

    assert 'endpoint="other"' in body
    assert 'endpoint="/v1/generate"' in body
