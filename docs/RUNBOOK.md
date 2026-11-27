# Project Nexus SRE Operational Runbook

This runbook maps directly to the Prometheus alerts in [`deploy/monitoring/prometheus-rules.yaml`](../deploy/monitoring/prometheus-rules.yaml) and internal admin endpoints (`/admin/fleet`, `/admin/usage`, `/metrics` on port `9090`).

## Quick Triage Commands

```bash
# Check pod liveness/readiness
curl -fsS https://<gateway>:8000/healthz
curl -fsS https://<gateway>:8000/readyz

# Inspect live heterogeneous fleet state (internal admin port 9090)
curl -fsS -H "Authorization: Bearer ${NEXUS_METRICS_BEARER_TOKEN}" \
  http://127.0.0.1:9090/admin/fleet

# Inspect org-wide tenant token/cost metering (internal admin port 9090)
curl -fsS -H "Authorization: Bearer ${NEXUS_METRICS_BEARER_TOKEN}" \
  http://127.0.0.1:9090/admin/usage
```

---

## Alert Playbooks

### `NexusGatewayHigh5xxRate`
1. Inspect structured JSON logs filtered by `status_code >= 500` and `outcome` (`timeout`, `backend_unavailable`, `backend_error`, `circuit_open`).
2. Query `/admin/fleet` to see which backend pool(s) show high EWMA latency, open circuits, or saturated KV-cache.
3. Verify upstream inference server pods (`kubectl get pods -n inference`) and GPU node health (`nvidia-smi` or `amd-smi` / `rocm-smi`).

### `NexusBackendCircuitOpen`
1. The circuit breaker opens after `circuit_breaker.failure_threshold` consecutive retryable failures and stays open for `cooldown_seconds` before sending a single `HALF_OPEN` probe.
2. Run `nexus probe --config /etc/nexus/nexus.yaml` from a debug container or check the upstream server `/health` endpoint directly.
3. If `routing.enable_failover: true` is configured with a healthy secondary pool (e.g., CUDA spillover from ROCm), confirm traffic is flowing cleanly to the secondary pool via `nexus_backend_requests_total`.

### `NexusBackendUnhealthy`
1. Check `GET /readyz` detail text (e.g., `ConnectError`, timeout, HTTP `503`).
2. Confirm upstream TLS certificates / mTLS client cert validity (`openssl x509 -noout -dates -in <cert>`).
3. Check Kubernetes `NetworkPolicy` egress rules if backend pod labels or ports changed.

### `NexusHighP95Latency`
1. Check `nexus_backend_kv_cache_usage_ratio` and `nexus_backend_queue_depth` in Grafana.
2. If upstream queue depth is climbing, scale the upstream GPU inference Deployment or enable `routing.strategy: least_inflight` / `cost_aware` across additional pools.
3. Verify `limits.max_tokens_limit` and `limits.max_prompt_chars` are aligned with upstream `--max-model-len`.

### `NexusGPUKVCacheSaturated`
1. Confirm `backend.metrics_path: "/metrics"` and `routing.telemetry_poll_interval_seconds` (e.g., `5.0`) are enabled so Nexus continuously scrapes upstream KV-cache ratios.
2. Verify secondary spillover pools have `kv_saturated: false` in `/admin/fleet`.
3. Scale upstream GPU replicas or lower `limits.max_concurrent_requests` to prevent OOM/preemption storms on the inference server.

### `NexusHighRateLimitRejections`
1. Separate `reason="rate_limit"` (per-caller RPM exceeded) from `reason="concurrency_limit"` (gateway active slot cap reached).
2. For multi-replica HA clusters, confirm `limits.rate_limit_backend: "redis"` is healthy and `NEXUS_LIMITS_RATE_LIMIT_REDIS_URL` is reachable.

### `NexusTenantQuotaExhausted`
1. Query `/admin/usage` to identify which `tenant_id` hit `max_tokens_per_window` or `max_cost_usd_per_window`.
2. Verify whether usage is legitimate burst traffic or a runaway client loop before raising tenant budgets.

---

## Standard Operational Procedures

### Zero-Downtime API Key Rotation
1. Add the new key to `NEXUS_AUTH_API_KEYS` (or a new tenant entry in `auth.named_api_keys`) while keeping the existing `NEXUS_AUTH_API_KEY` active.
2. Perform a rolling restart of `nexus-gateway` (`preStop` + `shutdown_drain_seconds` drains in-flight streams cleanly).
3. Migrate callers to the new token, verify traffic metrics, then remove the old key and roll again.

### TLS / mTLS Certificate Rotation
1. Mount TLS certificates via cert-manager / CSI Secret Store volume rather than baking them into container images.
2. Trigger a rolling restart (`kubectl rollout restart deployment/nexus-gateway`) after cert renewal; `PodDisruptionBudget` (`minAvailable: 1`) preserves availability during rotation.
