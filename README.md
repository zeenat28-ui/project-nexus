# Project Nexus

Project Nexus is a lightweight LLM serving integration and operations gateway written in Python (`v0.2.0`). It provides a consistent HTTP API (including SSE streaming and OpenAI-compatible chat endpoints), multi-backend model routing with per-backend circuit breakers and optional failover, health/readiness probes with graceful shutdown draining, validated configuration, rotatable API-key and OIDC/JWT authentication, in-memory or Redis cluster-wide rate limits, structured logs, optional OpenTelemetry OTLP tracing, and Prometheus metrics in front of external LLM inference servers.

## What Nexus Does — and Does Not Do

**What Nexus does:**
- Exposes consistent HTTP endpoints (`POST /v1/generate`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`, `GET /readyz`) with both JSON and SSE streaming (`stream: true`) responses.
- Ships with two backends implementing the `InferenceBackend` interface:
  - `mock`: deterministic CPU-only backend for local development, CI, and failure-mode testing.
  - `openai_compatible`: HTTP adapter for external inference servers exposing OpenAI-compatible APIs (such as vLLM, SGLang, llama.cpp server, or Text Generation Inference), including optional upstream mTLS client certificates.
- Routes requests across multiple named backend pools (`backends:` + `routing:`) by model name—for example, separate NVIDIA CUDA and AMD ROCm vLLM pools—each with its own isolated circuit breaker and opt-in failover (`routing.enable_failover`).
- Validates YAML/TOML and `NEXUS_*` environment configuration with strict schemas (`extra="forbid"`) and secret redaction.
- Supports rotatable static bearer tokens (`auth.api_key` / `auth.api_keys`) and asymmetric RS256/ES256 JWT verification via JWKS URL or static PEM public key (`auth.mode: jwt`).
- Enforces request body byte limits, prompt/token bounds, per-caller rate limits (in-memory bounded LRU or cluster-wide Redis Lua token bucket), max concurrent request caps, timeouts, client disconnect/cancellation handling, and graceful shutdown draining.
- Emits structured JSON/text logs with sanitized `X-Request-ID` correlation and optional OpenTelemetry OTLP traces (with W3C `traceparent` propagation) while excluding prompts, messages, and credentials from logs and spans.
- Exposes Prometheus metrics with strictly bounded label cardinality on a separate loopback/internal listener by default.
- Provides a read-only host GPU capability diagnostic (`nexus detect-gpu`), a staging verifier (`nexus verify-backend`), and a concurrent load benchmark (`nexus benchmark`).

**What Nexus does NOT do:**
- Nexus is **not** a GPU runtime, kernel library, or model execution engine. It does not replace CUDA, ROCm, PyTorch, vLLM, Triton, NCCL, or RCCL.
- Nexus does **not** implement custom VRAM memory pooling, custom weight hot-swapping, or tensor-parallel collectives. Memory allocation and KV-cache management belong in the upstream inference engine (for example, vLLM PagedAttention), where the runtime has direct visibility into model execution and allocator state.
- Nexus does **not** support mixed-vendor (NVIDIA + AMD) multi-GPU execution inside a single inference process. Multi-backend routing connects to separate vendor-specific upstream servers.
- Nexus does **not** provide built-in TLS termination or multi-tenant billing/metering.

## Architecture

```mermaid
flowchart LR
    Client["Client / OpenAI SDK / Ingress (TLS)"] -->|POST /v1/generate, /v1/chat/completions, GET /v1/models| Gateway["Project Nexus Gateway (FastAPI, port 8000)"]
    Probe["K8s Probes / Ops"] -->|GET /healthz, /readyz| Gateway
    Prom["Prometheus Scraper (Internal Network)"] -->|GET /metrics + Bearer Token (port 9090)| Metrics["Nexus Internal Metrics Server"]
    Gateway --- Metrics
    Gateway -.->|Optional Cluster Rate Limits| Redis["Redis (Atomic Lua Token Bucket)"]
    Gateway -.->|Optional OTLP/HTTP Spans| OTel["OpenTelemetry Collector"]
    Gateway -->|BackendRouter + Per-Pool Circuit Breakers| Mock["MockBackend (Local CPU / Tests)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE| CUDA["NVIDIA CUDA Pool (e.g. vLLM CUDA)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE| ROCm["AMD ROCm Pool (e.g. vLLM ROCm)"]
```

## Feature Support & Verification Matrix

| Capability | Status | Verification |
| --- | --- | --- |
| Python package (`project-nexus`), `uv.lock`, & `nexus` CLI | Implemented | Automated tests, `uv lock --check`, build & CLI smoke tests |
| FastAPI gateway (`/healthz`, `/readyz`, `/v1/generate`, `/v1/chat/completions`, `/v1/models`) | Implemented | Automated unit & ASGI integration tests |
| SSE streaming (`stream: true`) on `/v1/generate` & `/v1/chat/completions` | Implemented | Automated streaming, error-before-headers, & mid-stream error tests |
| Multi-backend model routing, per-pool circuit breakers, & opt-in failover | Implemented | Automated routing, failover, & isolated circuit-breaker tests |
| `MockBackend` (deterministic + failure simulation + streaming) | Implemented | Automated tests on CPU |
| `OpenAICompatibleBackend` (chat/completions, SSE, health, mTLS SSLContext) | Implemented | Automated tests via `httpx.MockTransport` & ephemeral PEM certs |
| Auth: rotatable API keys & asymmetric RS256/ES256 JWT/JWKS validation | Implemented | Automated tests with generated RSA/EC keys & forged `alg=none` rejection |
| Rate limiting (in-memory LRU + optional Redis Lua) & concurrency cap | Implemented | Automated unit, mock-Redis fail-open/closed, & HTTP `429` tests |
| Optional OpenTelemetry tracing (`[otel]`) with built-in OTLP/HTTP exporter | Implemented | Automated `InMemorySpanExporter`, OTLP config, & `traceparent` tests |
| Prometheus metrics (separate default listener, bounded labels) | Implemented | Automated scrape & cardinality tests |
| `nexus detect-gpu`, `nexus verify-backend`, & `nexus benchmark` | Implemented | Automated unit & CLI smoke tests |
| Live external NVIDIA CUDA server (vLLM on NVIDIA Tesla T4) | **Verified** | Verified end-to-end on Colab NVIDIA Tesla T4 (`vLLM 0.34.0`, `CUDA 13.0`, `Qwen/Qwen2.5-1.5B-Instruct`): health, non-streaming, SSE streaming, auth, & metrics ([details](#nvidia-cuda-upstream-backend-verified-on-tesla-t4)) |
| Live external AMD ROCm server (e.g. vLLM on ROCm) | Supported via `openai_compatible` | **Not yet verified on physical AMD hardware** (run [`scripts/verify_rocm_vllm.sh`](scripts/verify_rocm_vllm.sh) on an AMD ROCm node) |
| Mixed-vendor NVIDIA + AMD single-process execution | Not supported | Explicitly unsupported |

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

1. Validate config, run the verifier, and run a local concurrent benchmark:

```bash
nexus check-config --config config/nexus.mock.yaml
nexus probe --config config/nexus.mock.yaml
nexus verify-backend --config config/nexus.mock.yaml --iterations 3
nexus benchmark --config config/nexus.mock.yaml --requests 10 --concurrency 2 --stream
```

2. Start the gateway (`127.0.0.1:8000` for API traffic, `127.0.0.1:9090` for internal Prometheus metrics):

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

# SSE streaming /v1/generate
curl -sN -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Stream this response", "stream": true}'

# OpenAI-compatible /v1/chat/completions
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello Nexus"}], "max_tokens": 64}'

# Prometheus metrics (separate loopback port by default; :8000/metrics returns 404)
curl -s http://127.0.0.1:9090/metrics | grep "^nexus_"
```

## Connecting to Single or Multiple Real Backends

- **Single external backend:** Copy and edit [`config/nexus.openai-compat.example.yaml`](config/nexus.openai-compat.example.yaml).
- **Multiple backends (e.g., separate CUDA and ROCm vLLM pools):** Copy and edit [`config/nexus.multi-backend.example.yaml`](config/nexus.multi-backend.example.yaml), defining named pools under `backends:` and model-to-pool mappings under `routing:`.

Supply secrets via environment variables rather than editing them into YAML files:

```bash
export NEXUS_AUTH_API_KEY="replace-with-gateway-token"
# Optional comma-separated secondary keys for zero-downtime key rotation:
# export NEXUS_AUTH_API_KEYS="previous-token-during-rotation"
export NEXUS_BACKEND_API_KEY="replace-with-upstream-token-if-required"
export NEXUS_METRICS_BEARER_TOKEN="replace-with-metrics-scrape-token"

nexus check-config --config config/nexus.openai-compat.example.yaml
nexus verify-backend --config config/nexus.openai-compat.example.yaml --iterations 5
nexus serve --config config/nexus.openai-compat.example.yaml
```

For OIDC/JWT validation instead of static bearer keys:

```bash
export NEXUS_AUTH_MODE="jwt"
export NEXUS_AUTH_JWT_JWKS_URL="https://idp.example.internal/.well-known/jwks.json"
export NEXUS_AUTH_JWT_ISSUER="https://idp.example.internal"
export NEXUS_AUTH_JWT_AUDIENCE="project-nexus"
```

For cluster-wide Redis rate limiting (`pip install "project-nexus[redis]"`):

```bash
export NEXUS_LIMITS_RATE_LIMIT_BACKEND="redis"
export NEXUS_LIMITS_RATE_LIMIT_RPM="300"
export NEXUS_LIMITS_RATE_LIMIT_BURST="50"
export NEXUS_LIMITS_RATE_LIMIT_REDIS_URL="rediss://:password@redis.internal:6379/0"
```

For OTLP/HTTP trace export (`pip install "project-nexus[otel]"`):

```bash
export NEXUS_TRACING_ENABLED="true"
export NEXUS_TRACING_OTLP_ENDPOINT="https://otel-collector.monitoring.svc.cluster.local:4318/v1/traces"
export NEXUS_TRACING_SAMPLE_RATIO="0.25"
```

## NVIDIA CUDA & AMD ROCm Environment Notes

Project Nexus itself runs as a CPU-only Python gateway container and connects over HTTP(S)/mTLS to vendor-specific inference servers. Keep NVIDIA CUDA and AMD ROCm backends as **separate deployments** backed by upstream-supported runtime images.

### NVIDIA CUDA Upstream Backend (Verified on Tesla T4)

- Upstream reference: [vLLM OpenAI-Compatible Server (CUDA)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- Reproducible Colab verification notebook: [`notebooks/colab_cuda_verification.ipynb`](notebooks/colab_cuda_verification.ipynb)

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
- Typical pairing: run a pinned ROCm-compatible vLLM release matching your Instinct GPU architecture (for example, MI250X / MI300X) and host ROCm driver version, then run `VLLM_BASE_URL=http://127.0.0.1:8001 MODEL_ID=<model> bash scripts/verify_rocm_vllm.sh` to validate `nexus detect-gpu`, `nexus verify-backend`, `nexus benchmark --stream`, and authenticated `nexus serve` E2E.
- Hardware verification status: **Protocol integration (non-streaming + SSE streaming) is tested via `httpx.MockTransport`, and `amd-smi` / `rocm-smi` parsing is unit-tested with mocked outputs; physical AMD ROCm execution has not yet been recorded.**

## Configuration Reference

Precedence order (lowest to highest):
1. Built-in defaults
2. Configuration file (`--config <path>` or `NEXUS_CONFIG_FILE`; supports `.yaml`, `.yml`, `.toml`)
3. Environment variables (`NEXUS_*`)
4. Explicit CLI flags (`--host`, `--port` on `nexus serve`)

Unknown keys in configuration files fail validation (`extra="forbid"`). YAML is loaded strictly with `yaml.safe_load`.

| Setting Path | Environment Variable | Default | Notes |
| --- | --- | --- | --- |
| `server.host` | `NEXUS_SERVER_HOST` | `127.0.0.1` | Loopback by default; set `0.0.0.0` explicitly in containers |
| `server.port` | `NEXUS_SERVER_PORT` | `8000` | `1–65535` |
| `server.shutdown_drain_seconds` | `NEXUS_SERVER_SHUTDOWN_DRAIN_SECONDS` | `10.0` | Graceful drain wait window before closing backend pools |
| `backend.type` | `NEXUS_BACKEND_TYPE` | `mock` | `mock` or `openai_compatible` (single-backend mode) |
| `backend.model` | `NEXUS_BACKEND_MODEL` | `mock-llm-v1` (mock only) | Required when `backend.type=openai_compatible` |
| `backend.base_url` | `NEXUS_BACKEND_BASE_URL` | `null` | Required `http://` or `https://` URL for `openai_compatible` |
| `backend.api_style` | `NEXUS_BACKEND_API_STYLE` | `chat_completions` | `chat_completions` or `completions` |
| `backend.health_path` | `NEXUS_BACKEND_HEALTH_PATH` | `/health` | Relative path starting with `/` |
| `backend.api_key` | `NEXUS_BACKEND_API_KEY` | `null` | `SecretStr`; redacted in logs and `check-config` |
| `backend.verify_tls` | `NEXUS_BACKEND_VERIFY_TLS` | `true` | TLS verification enabled by default |
| `backend.ca_bundle_path` | `NEXUS_BACKEND_CA_BUNDLE_PATH` | `null` | Optional custom CA bundle path |
| `backend.client_cert_path` / `client_key_path` | `NEXUS_BACKEND_CLIENT_CERT_PATH` / `KEY_PATH` | `null` | Upstream mTLS client cert + key (must be paired) |
| `backends.<pool>.*` | *(config file)* | `{}` | Optional named pools for multi-backend model routing |
| `routing.default_backend` | `NEXUS_ROUTING_DEFAULT_BACKEND` | first pool | Default pool when `backends:` is configured |
| `routing.model_routes` | *(config file)* | `{}` | Map of `model_id -> pool_name` |
| `routing.enable_failover` | `NEXUS_ROUTING_ENABLE_FAILOVER` | `false` | Opt-in retryable failover across `fallback_order` |
| `auth.mode` | `NEXUS_AUTH_MODE` | `auto` | `auto`, `none`, `api_key`, or `jwt` |
| `auth.api_key` / `auth.api_keys` | `NEXUS_AUTH_API_KEY` / `NEXUS_AUTH_API_KEYS` | `null` / `[]` | Primary + CSV secondary rotatable bearer tokens |
| `auth.jwt.*` | `NEXUS_AUTH_JWT_*` | `null` | Asymmetric RS256/ES256 JWKS URL or PEM public key, `iss`, `aud` |
| `limits.max_request_bytes` | `NEXUS_LIMITS_MAX_REQUEST_BYTES` | `65536` | Enforced on `Content-Length` and streamed chunks (`413`) |
| `limits.max_prompt_chars` | `NEXUS_LIMITS_MAX_PROMPT_CHARS` | `16384` | Maximum prompt / combined message length (`422`) |
| `limits.max_tokens_limit` | `NEXUS_LIMITS_MAX_TOKENS_LIMIT` | `2048` | Upper cap on per-request `max_tokens` (`422`) |
| `limits.request_timeout_seconds` | `NEXUS_LIMITS_REQUEST_TIMEOUT_SECONDS` | `35.0` | End-to-end backend call timeout (`504`) |
| `limits.max_concurrent_requests` | `NEXUS_LIMITS_MAX_CONCURRENT_REQUESTS` | `64` | Fail-fast per-replica active inference cap (`429`) |
| `limits.rate_limit_backend` | `NEXUS_LIMITS_RATE_LIMIT_BACKEND` | `memory` | `memory` (per-replica LRU) or `redis` (cluster-wide Lua) |
| `limits.rate_limit_requests_per_minute` | `NEXUS_LIMITS_RATE_LIMIT_RPM` | `0` | Per-caller RPM (`0` disables rate limiting) |
| `limits.rate_limit_redis_url` | `NEXUS_LIMITS_RATE_LIMIT_REDIS_URL` | `null` | `SecretStr` `redis://` or `rediss://` URL (redacted) |
| `limits.rate_limit_redis_fail_open` | `NEXUS_LIMITS_RATE_LIMIT_REDIS_FAIL_OPEN` | `true` | Allow (`true`) or reject (`false`) if Redis is unreachable |
| `circuit_breaker.failure_threshold` / `cooldown_seconds` | `NEXUS_CIRCUIT_BREAKER_*` | `3` / `15.0` | Per-backend circuit breaker thresholds |
| `tracing.enabled` / `otlp_endpoint` / `sample_ratio` | `NEXUS_TRACING_*` | `false` / `null` / `1.0` | Optional OpenTelemetry spans + OTLP/HTTP export |
| `metrics.mode` / `host` / `port` / `bearer_token` | `NEXUS_METRICS_*` | `separate` / `127.0.0.1` / `9090` / `null` | Dedicated metrics listener by default |

## Docker, Kubernetes & Releases

- **Docker local dev:** `docker compose up --build` (binds only host `127.0.0.1:8000`, `read_only: true`, `cap_drop: [ALL]`).
- **Kubernetes base & HA manifests:**
  - Base: `kubectl kustomize deploy/kubernetes/base` (`ConfigMap`, non-root `Deployment`, `ClusterIP Service`, `NetworkPolicy`)
  - HA overlay: `kubectl kustomize deploy/kubernetes/overlays/ha` (adds `replicas: 2`, `topologySpreadConstraints`, `PodDisruptionBudget`, `HorizontalPodAutoscaler`)
  - Secret templates: [`deploy/kubernetes/secret.example.yaml`](deploy/kubernetes/secret.example.yaml) and [`deploy/kubernetes/externalsecret.example.yaml`](deploy/kubernetes/externalsecret.example.yaml)
- **Automated Releases & Supply Chain:** Pushing a semver tag (`git tag v0.2.0 && git push origin v0.2.0`) triggers [`.github/workflows/release.yml`](.github/workflows/release.yml) to build and push `ghcr.io/zeenat28-ui/project-nexus`, sign the immutable digest keyless with Sigstore Cosign, attach an SPDX SBOM, and publish Python wheel/sdist artifacts to GitHub Releases.

## Security, Contributing & License

- **Security policy & trust boundaries:** See [`SECURITY.md`](SECURITY.md).
- **Contributing guide:** See [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **License:** Licensed under the [Apache License, Version 2.0](LICENSE).
