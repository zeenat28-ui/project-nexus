# Project Nexus

[![CI](https://github.com/zeenat28-ui/project-nexus/actions/workflows/ci.yml/badge.svg)](https://github.com/zeenat28-ui/project-nexus/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Project Nexus is a lightweight LLM serving integration, fleet routing, and operations gateway written in Python (`v0.3.0`). It provides a consistent HTTP/HTTPS API (including SSE streaming and OpenAI-compatible chat endpoints), heterogeneous CUDA/ROCm multi-pool scheduling with upstream KV-cache pressure spillover, built-in TLS and inbound/upstream mTLS, multi-tenant token metering and USD budget enforcement, rotatable API-key and OIDC/JWT authentication, in-memory or Redis rate limits, structured logs, OpenTelemetry OTLP tracing, and Prometheus metrics in front of external LLM inference servers.

## What Nexus Does — and Intentional Architectural Boundaries

**What Nexus does:**
- Exposes consistent HTTP/HTTPS endpoints (`POST /v1/generate`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/usage`, `GET /healthz`, `GET /readyz`) with both JSON and SSE streaming (`stream: true`).
- Provides **built-in TLS termination and optional inbound mTLS** (`server.tls_cert_path`, `server.tls_key_path`, `server.tls_ca_bundle_path`, `server.tls_require_client_cert`) as well as **upstream mTLS** (`backend.client_cert_path`, `backend.client_key_path`).
- Orchestrates heterogeneous **NVIDIA CUDA, AMD ROCm, and CPU inference server pools** behind one model API using `priority`, `weighted`, `least_inflight`, `latency_ewma`, or `cost_aware` scheduling strategies, isolated per-pool circuit breakers, opt-in failover, and **upstream vLLM/SGLang KV-cache telemetry scraping** (`backend.metrics_path`) to spill over traffic away from saturated GPUs.
- Tracks **multi-tenant token usage, estimated USD cost, and windowed budgets** (`metering:` with optional SQLite persistence), enforcing `402 Payment Required` when a tenant exhausts its token or USD quota and exposing caller self-service (`GET /v1/usage`) plus authenticated internal admin summaries (`GET /admin/usage`, `GET /admin/fleet` on port `9090`).
- Supports named/rotatable static API keys (`auth.named_api_keys`, `auth.api_key`, `auth.api_keys`) and asymmetric `RS256`/`ES256` JWT/JWKS validation (`auth.mode: jwt`).
- Enforces body size caps (`413`), prompt/token limits (`422`), per-caller rate limits (in-memory LRU or cluster-wide Redis Lua), concurrency caps (`429`), timeouts (`504`), cancellation handling, and graceful shutdown draining.
- Emits structured JSON/text logs, optional OpenTelemetry OTLP/HTTP traces, and bounded-cardinality Prometheus metrics while never logging prompts, messages, or credentials.
- Ships with `nexus detect-gpu`, `nexus verify-backend`, and `nexus benchmark` for hardware capability diagnostics, staging validation, and concurrent load testing.

**Intentional architectural boundaries (what Nexus delegates to upstream runtimes):**
- Nexus is an **integration and fleet control plane**, not a replacement for CUDA, ROCm, PyTorch, vLLM, SGLang, Triton, NCCL, or RCCL.
- **VRAM / KV-cache allocation & tensor-parallel collectives belong in the inference runtime:** PagedAttention memory management and GPU-to-GPU collectives require direct kernel-level integration with the model executor. Nexus observes real KV-cache pressure from upstream engines via Prometheus scraping rather than reimplementing a C++ GPU allocator.
- **No single-process mixed NVIDIA + AMD GPU execution:** NVIDIA NCCL and AMD RCCL are hardware-incompatible within a single collective communicator. Nexus solves multi-vendor fleets cleanly by routing and load-balancing across separate, isolated CUDA and ROCm inference server pools.

## Architecture

```mermaid
flowchart LR
    Client["Client / OpenAI SDK (HTTPS / mTLS)"] -->|POST /v1/generate, /v1/chat/completions, GET /v1/models, /v1/usage| Gateway["Project Nexus Gateway (FastAPI + Built-in TLS, port 8000)"]
    Probe["K8s Probes / Ops"] -->|GET /healthz, /readyz| Gateway
    Admin["Prometheus & Internal Ops (Bearer Auth / TLS)"] -->|GET /metrics, /admin/usage, /admin/fleet (port 9090)| Internal["Nexus Internal Admin & Metrics Server"]
    Gateway --- Internal
    Gateway -.->|Optional Cluster Rate Limits| Redis["Redis (Atomic Lua Token Bucket)"]
    Gateway -.->|Optional Usage Ledger| SQLite["SQLite Metering Ledger"]
    Gateway -.->|Optional OTLP/HTTP Spans| OTel["OpenTelemetry Collector"]
    Gateway -->|Heterogeneous Scheduler + KV-Cache Spillover + Per-Pool Circuit Breakers| Mock["MockBackend (Local CPU / Tests)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE + /metrics scrape| CUDA["NVIDIA CUDA Pool (e.g. vLLM CUDA)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE + /metrics scrape| ROCm["AMD ROCm Pool (e.g. vLLM ROCm)"]
```

## Feature Support & Verification Matrix

| Capability | Status | Verification |
| --- | --- | --- |
| Python package (`project-nexus`), `uv.lock`, & `nexus` CLI | Implemented | Automated tests, `uv lock --check`, build & CLI smoke tests |
| Built-in HTTPS TLS termination & inbound client-cert mTLS | Implemented | Live HTTPS + mTLS E2E tests with ephemeral CA/server/client certs |
| FastAPI gateway (`/healthz`, `/readyz`, `/v1/generate`, `/v1/chat/completions`, `/v1/models`, `/v1/usage`) | Implemented | Automated unit & ASGI integration tests |
| SSE streaming (`stream: true`) on `/v1/generate` & `/v1/chat/completions` | Implemented | Automated streaming, error-before-headers, & mid-stream error tests |
| Heterogeneous fleet scheduler (`priority`, `weighted`, `least_inflight`, `latency_ewma`, `cost_aware`) + upstream vLLM/SGLang KV-cache spillover + `/admin/fleet` | Implemented | Automated multi-pool CUDA/ROCm scheduling, telemetry scrape, & spillover tests |
| Multi-tenant token/USD metering, SQLite ledger, `402` quotas, `/v1/usage`, & `/admin/usage` | Implemented | Automated persistence, window rollover, streaming metering, & `402` quota tests |
| Auth: named/rotatable API keys & asymmetric RS256/ES256 JWT/JWKS | Implemented | Automated RSA/EC key verification & forged `alg=none` rejection tests |
| Rate limiting (in-memory LRU + optional Redis Lua) & concurrency cap | Implemented | Automated unit, mock-Redis fail-open/closed, & HTTP `429` tests |
| Optional OpenTelemetry tracing (`[otel]`) with built-in OTLP/HTTP exporter | Implemented | Automated `InMemorySpanExporter`, OTLP config, & `traceparent` tests |
| Prometheus metrics (separate default listener, bounded labels) | Implemented | Automated scrape, token/cost metric, & cardinality tests |
| Docker build, non-root smoke test, Trivy SPDX SBOM, & `kubeconform -strict` (base + HA) | Implemented | Passed in GitHub Actions CI ([run #45947739159](https://github.com/zeenat28-ui/project-nexus/actions/runs/45947739159)) |
| Live external NVIDIA CUDA server (vLLM on NVIDIA Tesla T4) | **Verified** | Verified end-to-end on Colab NVIDIA Tesla T4 (`vLLM 0.34.0`, `CUDA 13.0`, `Qwen/Qwen2.5-1.5B-Instruct`): health, non-streaming, SSE streaming, auth, & metrics ([details](#nvidia-cuda-upstream-backend-verified-on-tesla-t4)) |
| Live external AMD ROCm server (e.g. vLLM on ROCm) | Supported via `openai_compatible` | **Not yet verified on physical AMD hardware** (run [`scripts/verify_rocm_vllm.sh`](scripts/verify_rocm_vllm.sh) on an AMD ROCm node) |

## Prerequisites

- Python **3.10+** (tested on Python 3.12 locally and 3.10–3.12 in CI)
- No GPU, CUDA toolkit, or ROCm installation is required to install, run locally with the mock backend, or execute the full test suite.
- Optional extras:
  - `pip install "project-nexus[otel]"` — OpenTelemetry SDK + OTLP/HTTP trace exporter
  - `pip install "project-nexus[redis]"` — Redis cluster-wide rate limiting

## Local Setup & Tests (CPU-Only)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"

# Run checks
uv lock --check
pytest --cov=project_nexus
ruff check src tests
ruff format --check src tests
mypy src
```

## Quickstart: End-to-End Mock Backend

1. Validate config, verify backend, and run a local concurrent benchmark:

```bash
nexus check-config --config config/nexus.mock.yaml
nexus probe --config config/nexus.mock.yaml
nexus verify-backend --config config/nexus.mock.yaml --iterations 3
nexus benchmark --config config/nexus.mock.yaml --requests 10 --concurrency 2 --stream
```

2. Start the gateway (`127.0.0.1:8000` for API traffic, `127.0.0.1:9090` for internal metrics/admin):

```bash
nexus serve --config config/nexus.mock.yaml
```

3. Test endpoints in a second terminal:

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/v1/models

# Non-streaming /v1/generate
curl -s -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-001" \
  -d '{"prompt": "Summarize Project Nexus", "max_tokens": 64, "temperature": 0.2}'

# SSE streaming /v1/chat/completions (OpenAI SDK compatible)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello Nexus"}], "stream": true}'

# Caller self-service usage summary
curl -s http://127.0.0.1:8000/v1/usage

# Internal metrics, org-wide usage, and fleet telemetry (port 9090; :8000/metrics returns 404)
curl -s http://127.0.0.1:9090/metrics | grep "^nexus_"
curl -s http://127.0.0.1:9090/admin/usage
curl -s http://127.0.0.1:9090/admin/fleet
```

## Connecting to Single or Heterogeneous CUDA/ROCm Backends

- **Single external backend:** Copy and edit [`config/nexus.openai-compat.example.yaml`](config/nexus.openai-compat.example.yaml).
- **Heterogeneous CUDA + ROCm fleet with KV-cache spillover & metering:** Copy and edit [`config/nexus.multi-backend.example.yaml`](config/nexus.multi-backend.example.yaml).

### Enabling Built-in TLS & Inbound mTLS

```bash
export NEXUS_SERVER_TLS_CERT_PATH="/etc/nexus/tls/server.crt"
export NEXUS_SERVER_TLS_KEY_PATH="/etc/nexus/tls/server.key"
# Optional inbound client-certificate verification (mTLS):
export NEXUS_SERVER_TLS_CA_BUNDLE_PATH="/etc/nexus/tls/client-ca.crt"
export NEXUS_SERVER_TLS_REQUIRE_CLIENT_CERT="true"
```

### Multi-Tenant Metering & Quotas

Configure named tenant keys (`auth.named_api_keys`) or JWT `org_id`/`sub` claims together with `metering:` pricing and per-window budgets (`max_tokens_per_window` / `max_cost_usd_per_window`). Exceeded budgets fail fast with HTTP `402 Payment Required` (`code: "quota_exceeded"`). Set `metering.sqlite_path` (`NEXUS_METERING_SQLITE_PATH`) to persist usage windows across restarts.

## NVIDIA CUDA & AMD ROCm Environment Notes

### NVIDIA CUDA Upstream Backend (Verified on Tesla T4)

- Upstream reference: [vLLM OpenAI-Compatible Server (CUDA)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- Reproducible Colab verification notebook: [`notebooks/colab_cuda_verification.ipynb`](notebooks/colab_cuda_verification.ipynb)
- Raw recorded JSON evidence: [`docs/evidence/nvidia-t4-colab-2026-11-27.json`](docs/evidence/nvidia-t4-colab-2026-11-27.json)

**Recorded single-GPU verification (2026-11-27, Google Colab Linux):**

| Field | Verified Result |
| --- | --- |
| GPU & Driver | NVIDIA Tesla T4 (15,360 MiB), Driver `580.82.07`, detected via `nexus detect-gpu --json` |
| Inference Stack | `vLLM 0.34.0`, `PyTorch 2.14.0+cu130`, `CUDA 13.0`, `Qwen/Qwen2.5-1.5B-Instruct` (`--dtype half --max-model-len 2048 --enforce-eager`) |
| `nexus verify-backend` | `overall_passed: true` — `/health` `59.97 ms`; 5 non-streaming requests `min 238.37 ms`, `p50 242.30 ms`, `p95/max 1245.18 ms` (first-request warmup); SSE streaming `ttft 107.02 ms`, `total 316.48 ms` (6 chunks) |
| Live Gateway E2E (`nexus serve`) | `/readyz` ready, `401` enforced on unauthenticated `/v1/models`, authenticated `/v1/generate` (`201.05 ms`), authenticated `/v1/chat/completions` (`stream: true` SSE + `[DONE]`), `:8000/metrics` `404`, `:9090/metrics` `200` with bearer token & zero prompt/request-ID leakage |

> Scope caveat: this validates single-GPU NVIDIA Tesla T4 integration with vLLM; multi-GPU tensor parallelism, high-concurrency saturation benchmarks, and newer NVIDIA architectures (L4/A100/H100) should still be benchmarked with `nexus benchmark` in your target staging cluster.

### AMD ROCm Upstream Backend (Unverified on Physical Hardware Yet)

- Upstream reference: [vLLM ROCm Installation & Serving Docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- Ready-to-run verification script for AMD Developer Cloud / Instinct nodes: [`scripts/verify_rocm_vllm.sh`](scripts/verify_rocm_vllm.sh)
- Typical pairing: run a pinned ROCm-compatible vLLM release matching your Instinct GPU architecture (for example, MI250X / MI300X) and host ROCm driver version, then run `VLLM_BASE_URL=http://127.0.0.1:8001 MODEL_ID=<model> bash scripts/verify_rocm_vllm.sh`.
- Hardware verification status: **Protocol integration, upstream KV-cache scraping, and `amd-smi`/`rocm-smi` parsing are unit-tested with mocks; physical AMD ROCm execution has not yet been recorded.**

## Configuration Highlights

Precedence (lowest to highest): **defaults < YAML/TOML file (`--config` / `NEXUS_CONFIG_FILE`) < `NEXUS_*` env vars < CLI flags**. Unknown keys are rejected (`extra="forbid"`); YAML uses `yaml.safe_load` only.

| Section | Key Settings / Env Vars | Default |
| --- | --- | --- |
| `server` | `host`, `port`, `shutdown_drain_seconds`, `tls_cert_path`, `tls_key_path`, `tls_ca_bundle_path`, `tls_require_client_cert` | `127.0.0.1:8000`, `10.0s`, TLS off |
| `backend` / `backends` | `type`, `base_url`, `model`, `api_style`, `health_path`, `metrics_path`, `max_kv_cache_usage_ratio`, `accelerator`, `gpu_count`, `weight`, `cost_tier`, `verify_tls`, `ca_bundle_path`, `client_cert_path`, `client_key_path` | `mock`, `verify_tls=true`, `0.95` KV threshold |
| `routing` | `default_backend`, `model_routes` (`str` or `list[str]`), `strategy` (`priority`/`weighted`/`least_inflight`/`latency_ewma`/`cost_aware`), `enable_failover`, `fallback_order` | `priority`, failover `false` |
| `auth` | `mode` (`auto`/`none`/`api_key`/`jwt`), `api_key`, `api_keys`, `named_api_keys`, `jwt.*` | `auto` |
| `metering` | `enabled`, `window_seconds`, `sqlite_path`, `model_pricing`, `default_max_tokens_per_window`, `default_max_cost_usd_per_window`, `tenants` | enabled, `86400s`, no cap |
| `limits` | `max_request_bytes`, `max_prompt_chars`, `max_tokens_limit`, `request_timeout_seconds`, `max_concurrent_requests`, `rate_limit_backend` (`memory`/`redis`), `rate_limit_requests_per_minute`, `rate_limit_redis_url` | `64 KiB`, `16384 chars`, `64` conc, `memory` |
| `tracing` | `enabled`, `service_name`, `otlp_endpoint`, `sample_ratio` | `false`, `1.0` |
| `metrics` | `mode` (`separate`/`main`/`disabled`), `host`, `port`, `bearer_token`, `tls_cert_path`, `tls_key_path` | `separate`, `127.0.0.1:9090` |

## Docker, Kubernetes & Releases

- **Docker local dev:** `docker compose up --build` (binds only host `127.0.0.1:8000`, `read_only: true`, `cap_drop: [ALL]`).
- **Kubernetes base & HA manifests:**
  - Base: `kubectl kustomize deploy/kubernetes/base` (`ConfigMap`, non-root `Deployment`, `ClusterIP Service`, `NetworkPolicy`)
  - HA overlay: `kubectl kustomize deploy/kubernetes/overlays/ha` (`replicas: 2`, `topologySpreadConstraints`, `PodDisruptionBudget`, `HorizontalPodAutoscaler`)
  - Secret templates: [`deploy/kubernetes/secret.example.yaml`](deploy/kubernetes/secret.example.yaml) and [`deploy/kubernetes/externalsecret.example.yaml`](deploy/kubernetes/externalsecret.example.yaml)
- **Automated Releases:** Pushing a semver tag (`git tag v0.3.0 && git push origin v0.3.0`) triggers [`.github/workflows/release.yml`](.github/workflows/release.yml) to build and push `ghcr.io/zeenat28-ui/project-nexus`, sign the immutable digest keyless with Sigstore Cosign, attach an SPDX SBOM, and publish Python wheel/sdist artifacts to GitHub Releases.

## Security, Contributing & License

- **Security policy & trust boundaries:** See [`SECURITY.md`](SECURITY.md).
- **Contributing guide:** See [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **License:** Licensed under the [Apache License, Version 2.0](LICENSE).
