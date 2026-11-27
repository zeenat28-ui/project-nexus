"""Tests for multi-tenant metering, cost accounting, SQLite persistence, and 402 quotas."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from project_nexus.app import create_app, create_metrics_app
from project_nexus.config import load_settings
from project_nexus.metering import ModelPricing, TenantBudget, UsageMeter
from project_nexus.metrics import NexusMetrics


def test_usage_meter_sqlite_persistence_and_window_rollover(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite3"
    now = [1_700_000_000.0]

    meter = UsageMeter(
        window_seconds=3600,
        model_pricing={
            "m1": ModelPricing(prompt_cost_per_1k_usd=0.5, completion_cost_per_1k_usd=1.0)
        },
        tenant_budgets={
            "team-a": TenantBudget(max_tokens_per_window=200, max_cost_usd_per_window=0.20)
        },
        sqlite_path=db_path,
        clock=lambda: now[0],
    )
    rec = meter.record_usage(
        tenant_id="team-a", model="m1", prompt_tokens=100, completion_tokens=50
    )
    assert rec.total_tokens == 150
    assert rec.estimated_cost_usd == 0.1
    assert meter.check_quota("team-a").allowed is True
    meter.close()

    # Reopen from SQLite and verify previous usage persists
    reopened = UsageMeter(
        window_seconds=3600,
        model_pricing={
            "m1": ModelPricing(prompt_cost_per_1k_usd=0.5, completion_cost_per_1k_usd=1.0)
        },
        tenant_budgets={
            "team-a": TenantBudget(max_tokens_per_window=200, max_cost_usd_per_window=0.20)
        },
        sqlite_path=db_path,
        clock=lambda: now[0],
    )
    summary = reopened.get_tenant_summary("team-a")
    assert summary["total_tokens"] == 150
    assert summary["estimated_cost_usd"] == 0.1
    assert summary["limits"]["remaining_tokens"] == 50

    # Pushing over token budget trips quota
    reopened.record_usage(tenant_id="team-a", model="m1", prompt_tokens=40, completion_tokens=20)
    denied = reopened.check_quota("team-a")
    assert denied.allowed is False
    assert denied.reason == "token_budget"

    # Advancing into next window resets budget
    now[0] += 3601.0
    assert reopened.check_quota("team-a").allowed is True
    reopened.close()


def test_http_multi_tenant_metering_quotas_streaming_and_admin_usage(tmp_path: Path) -> None:
    cfg_path = tmp_path / "metering.yaml"
    cfg_path.write_text(
        f"""
auth:
  mode: "api_key"
  named_api_keys:
    team-alpha: "alpha-secret-token"
    team-beta: "beta-secret-token"

metering:
  enabled: true
  window_seconds: 3600
  sqlite_path: "{(tmp_path / "ledger.db").as_posix()}"
  default_prompt_cost_per_1k_usd: 1.0
  default_completion_cost_per_1k_usd: 2.0
  tenants:
    team-alpha:
      max_tokens_per_window: 5
    team-beta:
      max_cost_usd_per_window: 10.0
""",
        encoding="utf-8",
    )
    settings = load_settings(cfg_path, env={})
    sanitized = settings.to_sanitized_dict()
    assert sanitized["auth"]["named_api_keys"]["team-alpha"] == "***REDACTED***"
    assert "alpha-secret-token" not in repr(settings)

    metrics = NexusMetrics()
    app = create_app(settings, metrics=metrics, configure_loggers=False)
    admin_app = create_metrics_app(
        metrics,
        bearer_token="admin-secret",
        usage_meter=app.state.usage_meter,
        router=app.state.router,
    )

    with TestClient(app) as client, TestClient(admin_app) as admin:
        # First request for team-alpha succeeds and consumes >=5 tokens
        ok1 = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer alpha-secret-token"},
            json={"prompt": "one two three four five"},
        )
        assert ok1.status_code == 200

        # Second request for team-alpha is blocked with 402 Payment Required
        blocked = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer alpha-secret-token"},
            json={"prompt": "over budget"},
        )
        assert blocked.status_code == 402
        assert blocked.json()["error"]["code"] == "quota_exceeded"
        assert "retry-after" in blocked.headers

        # team-beta is completely unaffected and streams cleanly
        stream_beta = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer beta-secret-token"},
            json={"prompt": "hello from beta", "stream": True},
        )
        assert stream_beta.status_code == 200
        assert "data: [DONE]" in stream_beta.text

        # Caller self-service /v1/usage shows only that tenant's usage
        alpha_usage = client.get(
            "/v1/usage", headers={"Authorization": "Bearer alpha-secret-token"}
        )
        assert alpha_usage.status_code == 200
        assert alpha_usage.json()["tenant_id"] == "team-alpha"
        assert alpha_usage.json()["requests"] == 1
        assert alpha_usage.json()["limits"]["remaining_tokens"] == 0

        # Internal /admin/usage requires bearer auth and shows org-wide breakdown
        assert admin.get("/admin/usage").status_code == 401
        org_usage = admin.get("/admin/usage", headers={"Authorization": "Bearer admin-secret"})
        assert org_usage.status_code == 200
        org_data = org_usage.json()
        assert org_data["totals"]["requests"] == 2
        assert "team-alpha" in org_data["tenants"]
        assert "team-beta" in org_data["tenants"]
        assert org_data["tenants"]["team-beta"]["total_tokens"] > 0

        # Prometheus metrics record tokens, cost, and quota rejection
        prom = admin.get("/metrics", headers={"Authorization": "Bearer admin-secret"}).text
        assert "nexus_tokens_total" in prom
        assert "nexus_estimated_cost_usd_total" in prom
        assert (
            'nexus_quota_rejected_total{endpoint="/v1/generate",reason="token_budget"} 1.0' in prom
        )
