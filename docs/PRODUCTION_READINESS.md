# Production Readiness Checklist & Verification Status

This document candidly separates what is **implemented and independently verified in Project Nexus `v0.4.0`** from **environment-specific gates** that each adopting organization must complete on its own infrastructure before serving critical production traffic.

## 1. Verified in Repository & CI (`v0.4.0`)

| Area | Verification Evidence |
| --- | --- |
| Core API, SSE streaming, & OpenAI compatibility | Automated ASGI tests + live NVIDIA Tesla T4 vLLM E2E (`docs/evidence/nvidia-t4-colab-2026-11-27.json`) |
| Built-in HTTPS TLS & inbound client-cert mTLS | Live TLS + mTLS handshake tests with ephemeral CA/server/client certificates (`tests/test_tls.py`) |
| Auth (named/rotatable API keys + asymmetric RS256/ES256 JWT/JWKS) | Automated verification + forged `alg=none` rejection tests (`tests/test_auth.py`) |
| Multi-tenant metering, `402` quotas, SQLite & Redis ledgers | Automated persistence, Redis Lua mock, and concurrent multi-tenant stress tests (`tests/test_metering.py`, `tests/test_production_stress.py`) |
| Heterogeneous CUDA + ROCm fleet routing & KV-cache spillover | Automated weighted/least-inflight/EWMA/cost-aware + vLLM/SGLang Prometheus scrape tests (`tests/test_fleet_routing.py`) |
| Concurrency, chaos failover, & security regression under load | Automated high-concurrency stress + chaos failover tests (`tests/test_production_stress.py`) + live localhost TCP gateway load benchmark (`docs/evidence/gateway-concurrency-benchmark-2026-11-27.json`) |
| Static & dependency security checks | `ruff` (with Bandit `S` rules), `bandit -r src`, `pip-audit`, `uv lock --check`, and Trivy container SPDX SBOM scan in CI |
| Container & Kubernetes HA packaging | Non-root `Dockerfile`, base + HA Kustomize overlays (`HPA`, `PDB`, `NetworkPolicy`, `preStop` drain), validated via `kubeconform -strict` in CI |
| Operational observability | Prometheus `ServiceMonitor`, 7 alert rules (`deploy/monitoring/prometheus-rules.yaml`), Grafana dashboard (`deploy/monitoring/grafana-dashboard.json`), and SRE runbook (`docs/RUNBOOK.md`) |

## 2. Target-Cluster Sign-Off Gates (Required Before Critical Production Traffic)

Organizations deploying Project Nexus for business-critical workloads must execute these environment-specific gates on their own infrastructure:

1. **Physical Multi-GPU & Target Accelerator Benchmarks:**
   - Run `nexus verify-backend` and `nexus benchmark --stream` against your actual production GPU fleet (for example, NVIDIA L4/A100/H100 or AMD Instinct MI250X/MI300X running pinned vLLM builds).
   - **AMD ROCm physical hardware:** Execute `scripts/verify_rocm_vllm.sh` on a physical AMD ROCm node; until recorded, AMD ROCm remains verified only via unit/protocol tests.
2. **Sustained Staging Load & Chaos Soak Test:**
   - Run `nexus load-test --url https://<staging-gateway> --requests 1000 --concurrency 50 --stream` against a multi-replica staging deployment backed by real Redis (`rate_limit_backend: redis`, `metering.backend: redis`) and live GPU backends while injecting pod restarts and upstream slowdowns.
3. **Immutable Image Release & Signing:**
   - Tag a release (`git tag v0.4.0 && git push origin v0.4.0`) to publish the signed GHCR container image (`release.yml`), verify its Cosign signature, and pin `deploy/kubernetes/base/deployment.yaml` to the immutable `@sha256:...` digest.
4. **External Security Review & Compliance:**
   - Review Ingress/service-mesh TLS, IdP JWKS availability, secret store policies (`ExternalSecret`), log retention policies, and any regulatory/compliance controls specific to your organization.
