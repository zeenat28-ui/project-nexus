# Project Nexus

Project Nexus is a lightweight LLM serving integration and operations gateway written in Python. It provides a consistent HTTP API (including SSE streaming and OpenAI-compatible chat endpoints), health/readiness probes with graceful shutdown draining, validated configuration, rotatable API-key and OIDC/JWT authentication, per-caller rate/concurrency limits, structured logs, optional OpenTelemetry tracing, circuit-breaker isolation, and Prometheus metrics in front of external LLM inference servers.

## What Nexus Does — and Does Not Do

**What Nexus does:**
- Exposes consistent HTTP endpoints (`POST /v1/generate`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`, `GET /readyz`) with both JSON and SSE streaming (`stream: true`) responses.
- Ships with two backends implementing the `InferenceBackend` interface:
  - `mock`: deterministic CPU-only backend for local development, CI, and failure-mode testing.
  - `openai_compatible`: HTTP adapter for external inference servers exposing OpenAI-compatible APIs (such as vLLM, SGLang, llama.cpp server, or Text Generation Inference), including optional upstream mTLS client certificates.
- Validates YAML/TOML and `NEXUS_*` environment configuration with strict schemas (`extra="forbid"`) and secret redaction.
- Supports rotatable static bearer tokens (`auth.api_key` / `auth.api_keys`) and asymmetric RS256/ES256 JWT verification via JWKS URL or static PEM public key (`auth.mode: jwt`).
- Enforces request body byte limits, prompt/token bounds, per-caller token-bucket rate limits, max concurrent request caps, timeouts, client disconnect/cancellation handling, graceful shutdown draining, and a backend circuit breaker.
- Emits structured JSON/text logs with sanitized `X-Request-ID` correlation and optional OpenTelemetry spans (with W3C `traceparent` propagation) while excluding prompts, messages, and credentials from logs and spans.
- Exposes Prometheus metrics with strictly bounded label cardinality on a separate loopback/internal listener by default.
- Provides a read-only host GPU capability diagnostic (`nexus detect-gpu`) and a repeatable staging verification harness (`nexus verify-backend`).

**What Nexus does NOT do:**
- Nexus is **not** a GPU runtime, kernel library, or model execution engine. It does not replace CUDA, ROCm, PyTorch, vLLM, Triton, NCCL, or RCCL.
- Nexus does **not** implement custom VRAM memory pooling, custom weight hot-swapping, or tensor-parallel collectives. Memory allocation and KV-cache management belong in the upstream inference engine (for example, vLLM PagedAttention), where the runtime has direct visibility into model execution and allocator state.
- Nexus does **not** support mixed-vendor (NVIDIA + AMD) multi-GPU execution.
- Nexus does **not** provide built-in TLS termination, multi-tenant billing, or globally synchronized multi-replica rate limits (built-in rate/concurrency limits are per-replica).

## Architecture

```mermaid
flowchart LR
    Client["Client / OpenAI SDK / Ingress (TLS)"] -->|POST /v1/generate, /v1/chat/completions, GET /v1/models| Gateway["Project Nexus Gateway (FastAPI, port 8000)"]
    Probe["K8s Probes / Ops"] -->|GET /healthz, /readyz| Gateway
    Prom["Prometheus Scraper (Internal Network)"] -->|GET /metrics + Bearer Token (port 9090)| Metrics["Nexus Internal Metrics Server"]
    Gateway --- Metrics
    Gateway -->|InferenceBackend Interface| Mock["MockBackend (Local CPU / Tests)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE| CUDA["External CUDA Server (e.g. vLLM on NVIDIA)"]
    Gateway -->|OpenAI-Compatible HTTP/mTLS + SSE| ROCm["External ROCm Server (e.g. vLLM on AMD)"]
```

## Feature Support & Verification Matrix

| Capability | Status | Verification |
| --- | --- | --- |
| Python package (`project-nexus`), `uv.lock`, & `nexus` CLI | Implemented | Automated tests, `uv lock --check`, build & CLI smoke tests |
| FastAPI gateway (`/healthz`, `/readyz`, `/v1/generate`, `/v1/chat/completions`, `/v1/models`) | Implemented | Automated unit & ASGI integration tests |
| SSE streaming (`stream: true`) on `/v1/generate` & `/v1/chat/completions` | Implemented | Automated streaming, error-before-headers, & mid-stream error tests |
| `MockBackend` (deterministic + failure simulation + streaming) | Implemented | Automated tests on CPU |
| `OpenAICompatibleBackend` (chat/completions, SSE, health, mTLS SSLContext) | Implemented | Automated tests via `httpx.MockTransport` & ephemeral PEM certs |
| Auth: rotatable API keys & asymmetric RS256/ES256 JWT/JWKS validation | Implemented | Automated tests with generated RSA/EC keys & forged `alg=none` rejection |
| Per-replica token-bucket rate limiter, concurrency cap, & graceful shutdown draining | Implemented | Automated unit & HTTP `429`/`503` drain tests |
| Optional OpenTelemetry tracing (`[otel]`) with strict attribute allowlist | Implemented | Automated `InMemorySpanExporter` & W3C `traceparent` tests |
| Prometheus metrics (separate default listener, bounded labels) | Implemented | Automated scrape & cardinality tests |
| `nexus detect-gpu` (`nvidia-smi`, `amd-smi`, `rocm-smi`) & `nexus verify-backend` | Implemented | Automated mocked GPU & mock/transport backend verification tests |
| Live external NVIDIA CUDA server (e.g. vLLM on CUDA) | Supported via `openai_compatible` | **Not verified on real NVIDIA hardware in this repo** |
| Live external AMD ROCm server (e.g. vLLM on ROCm) | Supported via `openai_compatible` | **Not verified on real AMD hardware in this repo** |
| Mixed-vendor NVIDIA + AMD execution | Not supported | Explicitly unsupported |

## Prerequisites

- Python **3.10+** (tested on Python 3.12 locally and 3.10–3.12 in CI)
- No GPU, CUDA toolkit, or ROCm installation is required to install, run locally with the mock backend, or execute the full test suite.
- Optional: Docker / Docker Compose (for containerized runs) and Kubernetes / Kustomize (for applying the manifests under `deploy/kubernetes/`).

## Local Setup & Tests (CPU-Only)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"

# Run checks
pytest --cov=project_nexus
ruff check src tests
ruff format --check src tests
mypy src
```

## Quickstart: End-to-End Mock Backend

1. Validate config and run the local verification harness:

```bash
nexus check-config --config config/nexus.mock.yaml
nexus probe --config config/nexus.mock.yaml
nexus verify-backend --config config/nexus.mock.yaml --iterations 3
```

2. Start the gateway (binds `127.0.0.1:8000` for API traffic and `127.0.0.1:9090` for internal Prometheus metrics):

```bash
nexus serve --config config/nexus.mock.yaml
```

3. In a second terminal, test health, readiness, non-streaming, SSE streaming, OpenAI chat completions, and metrics:

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

## Connecting to a Real Inference Backend (`openai_compatible`)

Nexus delegates GPU execution to an external inference server that speaks the OpenAI HTTP API.

1. Start your inference server separately (for example, vLLM listening on `http://127.0.0.1:8001`).
2. Copy and adjust `config/nexus.openai-compat.example.yaml`, setting `backend.base_url`, `backend.model`, `backend.api_style` (`chat_completions` or `completions`), and `backend.health_path` (`/health` for vLLM/SGLang or `/v1/models` for servers without `/health`).
3. Supply secrets via environment variables rather than editing them into YAML files:

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

For upstream mTLS between Nexus and the inference server, set `backend.ca_bundle_path`, `backend.client_cert_path`, and `backend.client_key_path` (certificate and key must be configured as a pair).

## NVIDIA CUDA & AMD ROCm Environment Notes

Project Nexus itself runs as a CPU-only Python gateway container and connects over HTTP(S)/mTLS to vendor-specific inference servers. Keep NVIDIA CUDA and AMD ROCm backends as **separate deployments** backed by upstream-supported runtime images.

Inspect local host GPU visibility and run repeatable staging smoke checks with:

```bash
nexus detect-gpu --json
nexus verify-backend --config <your-staging-config.yaml> --iterations 5
```

`nexus detect-gpu` only reports a vendor as `available: true` when `nvidia-smi` (NVIDIA) or `amd-smi` / `rocm-smi` (AMD) executes cleanly and returns parseable device entries. Presence of `/opt/rocm` or `ROCM_PATH` alone is never treated as a verified GPU runtime.

### NVIDIA CUDA Upstream Backend (Unverified on Hardware Here)

- Upstream reference: [vLLM OpenAI-Compatible Server (CUDA)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- Ready-to-run Colab verification notebook: [`notebooks/colab_cuda_verification.ipynb`](notebooks/colab_cuda_verification.ipynb) (runs on Google Colab NVIDIA T4 / L4 / A100 runtimes, starts vLLM CUDA with `Qwen/Qwen2.5-1.5B-Instruct`, and executes `nexus detect-gpu`, `nexus verify-backend`, and a live authenticated SSE gateway check).
- Typical pairing: run a pinned vLLM CUDA release on your NVIDIA GPU host/node with `--host 127.0.0.1 --port 8001 --model <model-id>`, then configure Nexus with `backend.type: openai_compatible`, `backend.base_url: http://127.0.0.1:8001`, `backend.health_path: /health`, and validate with `nexus verify-backend`.
- Hardware verification status: **Protocol integration (non-streaming + SSE streaming) is tested via `httpx.MockTransport`, and `nvidia-smi` parsing is unit-tested with mocked outputs; end-to-end execution on physical NVIDIA GPUs has not yet been recorded in this repository.**

### AMD ROCm Upstream Backend (Unverified on Hardware Here)

- Upstream reference: [vLLM ROCm Installation & Serving Docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- Typical pairing: run a pinned ROCm-compatible vLLM release matching your Instinct GPU architecture (for example, MI250X / MI300X) and host ROCm driver version, expose its OpenAI-compatible HTTP endpoint on an internal network, and validate with `nexus verify-backend`.
- Hardware verification status: **Protocol integration (non-streaming + SSE streaming) is tested via `httpx.MockTransport`, and `amd-smi` / `rocm-smi` parsing is unit-tested with mocked outputs; end-to-end execution on physical AMD GPUs has not been executed in this repository.**

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
| `server.shutdown_drain_seconds` | `NEXUS_SERVER_SHUTDOWN_DRAIN_SECONDS` | `10.0` | Graceful drain wait window before closing backend pool |
| `backend.type` | `NEXUS_BACKEND_TYPE` | `mock` | `mock` or `openai_compatible` |
| `backend.model` | `NEXUS_BACKEND_MODEL` | `mock-llm-v1` (mock only) | Required when `backend.type=openai_compatible` |
| `backend.base_url` | `NEXUS_BACKEND_BASE_URL` | `null` | Required `http://` or `https://` URL for `openai_compatible` |
| `backend.api_style` | `NEXUS_BACKEND_API_STYLE` | `chat_completions` | `chat_completions` or `completions` |
| `backend.health_path` | `NEXUS_BACKEND_HEALTH_PATH` | `/health` | Relative path starting with `/` |
| `backend.api_key` | `NEXUS_BACKEND_API_KEY` | `null` | `SecretStr`; redacted in logs and `check-config` |
| `backend.connect_timeout_seconds` | `NEXUS_BACKEND_CONNECT_TIMEOUT_SECONDS` | `5.0` | `> 0`, `<= 60` |
| `backend.read_timeout_seconds` | `NEXUS_BACKEND_READ_TIMEOUT_SECONDS` | `30.0` | `> 0`, `<= 300` |
| `backend.verify_tls` | `NEXUS_BACKEND_VERIFY_TLS` | `true` | TLS verification enabled by default |
| `backend.ca_bundle_path` | `NEXUS_BACKEND_CA_BUNDLE_PATH` | `null` | Optional custom CA bundle path |
| `backend.client_cert_path` | `NEXUS_BACKEND_CLIENT_CERT_PATH` | `null` | Upstream mTLS client certificate (paired with key) |
| `backend.client_key_path` | `NEXUS_BACKEND_CLIENT_KEY_PATH` | `null` | Upstream mTLS client private key (paired with cert) |
| `backend.max_connections` | `NEXUS_BACKEND_MAX_CONNECTIONS` | `100` | HTTP connection pool max |
| `backend.max_keepalive_connections` | `NEXUS_BACKEND_MAX_KEEPALIVE_CONNECTIONS` | `20` | HTTP keepalive pool max |
| `backend.mock.latency_ms` | `NEXUS_MOCK_LATENCY_MS` | `0.0` | Simulated mock delay (`0–30000`) |
| `backend.mock.mode` | `NEXUS_MOCK_MODE` | `normal` | `normal`, `timeout`, `unavailable`, `error`, `unhealthy` |
| `backend.mock.response_prefix` | `NEXUS_MOCK_RESPONSE_PREFIX` | `[mock]` | Deterministic prefix for mock output |
| `auth.mode` | `NEXUS_AUTH_MODE` | `auto` | `auto`, `none`, `api_key`, or `jwt` |
| `auth.api_key` | `NEXUS_AUTH_API_KEY` | `null` | Primary bearer token for `/v1/*` endpoints |
| `auth.api_keys` | `NEXUS_AUTH_API_KEYS` | `[]` | Comma-separated additional tokens for key rotation |
| `auth.jwt.jwks_url` | `NEXUS_AUTH_JWT_JWKS_URL` | `null` | OIDC JWKS endpoint for RS256/ES256 verification |
| `auth.jwt.public_key_pem` | `NEXUS_AUTH_JWT_PUBLIC_KEY_PEM` | `null` | Static PEM public key alternative to `jwks_url` |
| `auth.jwt.issuer` | `NEXUS_AUTH_JWT_ISSUER` | `null` | Expected JWT `iss` claim |
| `auth.jwt.audience` | `NEXUS_AUTH_JWT_AUDIENCE` | `null` | Expected JWT `aud` claim |
| `limits.max_request_bytes` | `NEXUS_LIMITS_MAX_REQUEST_BYTES` | `65536` | Enforced on `Content-Length` and streamed chunks (`413`) |
| `limits.max_prompt_chars` | `NEXUS_LIMITS_MAX_PROMPT_CHARS` | `16384` | Maximum prompt / combined message character length (`422`) |
| `limits.max_tokens_limit` | `NEXUS_LIMITS_MAX_TOKENS_LIMIT` | `2048` | Upper cap on per-request `max_tokens` (`422`) |
| `limits.request_timeout_seconds` | `NEXUS_LIMITS_REQUEST_TIMEOUT_SECONDS` | `35.0` | End-to-end backend call timeout (`504`) |
| `limits.health_check_timeout_seconds` | `NEXUS_LIMITS_HEALTH_CHECK_TIMEOUT_SECONDS` | `3.0` | `/readyz` upstream health timeout (`503`) |
| `limits.max_concurrent_requests` | `NEXUS_LIMITS_MAX_CONCURRENT_REQUESTS` | `64` | Fail-fast per-replica active inference cap (`429`) |
| `limits.rate_limit_requests_per_minute` | `NEXUS_LIMITS_RATE_LIMIT_RPM` | `0` | Per-caller per-replica RPM (`0` disables rate limiting) |
| `limits.rate_limit_burst` | `NEXUS_LIMITS_RATE_LIMIT_BURST` | `null` | Token bucket burst size (defaults to RPM when enabled) |
| `circuit_breaker.failure_threshold` | `NEXUS_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive retryable backend failures before opening |
| `circuit_breaker.cooldown_seconds` | `NEXUS_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `15.0` | Seconds before transitioning `OPEN` -> `HALF_OPEN` |
| `logging.level` | `NEXUS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `logging.format` | `NEXUS_LOG_FORMAT` | `json` | `json` or `text` |
| `tracing.enabled` | `NEXUS_TRACING_ENABLED` | `false` | Enables optional OpenTelemetry spans (`pip install .[otel]`) |
| `tracing.service_name` | `NEXUS_TRACING_SERVICE_NAME` | `project-nexus` | OTel tracer service name |
| `metrics.mode` | `NEXUS_METRICS_MODE` | `separate` | `separate`, `main`, or `disabled` |
| `metrics.host` | `NEXUS_METRICS_HOST` | `127.0.0.1` | Bind host for `separate` metrics listener |
| `metrics.port` | `NEXUS_METRICS_PORT` | `9090` | Bind port for `separate` metrics listener |
| `metrics.bearer_token` | `NEXUS_METRICS_BEARER_TOKEN` | `null` | Required if `metrics.mode=main` (unless opted out) |
| `metrics.allow_unauthenticated_main` | `NEXUS_METRICS_ALLOW_UNAUTHENTICATED_MAIN` | `false` | Explicit opt-in required to expose unauthenticated `/metrics` on main port |

## Docker Local Development

Build and run the CPU-only gateway image with Docker:

```bash
docker build -t project-nexus:0.1.0-local .
docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp:size=64m project-nexus:0.1.0-local
```

Or use Docker Compose (binds to host `127.0.0.1:8000` with `read_only: true`, `cap_drop: [ALL]`, and `no-new-privileges:true`):

```bash
docker compose up --build
```

## Kubernetes Deployment Examples & Security Caveats

Hardened base manifests live in [`deploy/kubernetes/base/`](deploy/kubernetes/base/):
- `configmap.yaml` — non-secret gateway configuration (including rate/concurrency limits and drain timeout)
- `deployment.yaml` — non-root (`10001`), `readOnlyRootFilesystem: true`, dropped Linux capabilities, `RuntimeDefault` seccomp profile, `preStop` drain sleep, `terminationGracePeriodSeconds: 30`, resource requests/limits, startup/liveness/readiness probes, and `automountServiceAccountToken: false`
- `service.yaml` — `ClusterIP` exposing only port `80` -> `8000` (excludes port `9090`)
- `networkpolicy.yaml` — restricts ingress on `8000` (clients) and `9090` (monitoring) and egress to DNS + inference backend
- `kustomization.yaml` — renders base manifests (`kubectl kustomize deploy/kubernetes/base`)

Secret templates and HA overlay:
- `deploy/kubernetes/secret.example.yaml` — template showing required secret keys (**never commit real values**)
- `deploy/kubernetes/externalsecret.example.yaml` — External Secrets Operator (ESO) template for Vault/AWS/GCP secret stores (requires ESO CRDs)
- [`deploy/kubernetes/overlays/ha/`](deploy/kubernetes/overlays/ha/) — multi-replica overlay adding `replicas: 2`, `topologySpreadConstraints`, `PodDisruptionBudget` (`minAvailable: 1`), and `HorizontalPodAutoscaler` (`minReplicas: 2`, `maxReplicas: 10`) (`kubectl kustomize deploy/kubernetes/overlays/ha`)

### Network Trust Boundaries & Operational Limitations

1. **TLS termination:** Nexus listens over plaintext HTTP. Terminate TLS at an Ingress controller, API gateway, or mTLS service mesh before exposing traffic outside localhost. For Nexus -> inference backend traffic, enable HTTPS (`verify_tls: true`) and optionally configure mTLS client certificates.
2. **Per-replica rate limits:** Built-in token-bucket rate limiting and concurrency caps operate in-memory per gateway replica. For globally synchronized multi-replica quotas or tenant billing, enforce quotas at an edge API gateway or shared store.
3. **Secrets management:** `deploy/kubernetes/secret.example.yaml` and `externalsecret.example.yaml` are templates excluded from `kustomization.yaml`. Provision secrets via External Secrets Operator, Vault, or KMS-backed secret stores.
4. **Container image pinning & signing:** `deployment.yaml` references `ghcr.io/zeenat28-ui/project-nexus:0.1.0` as an example tag. Build, scan (Trivy/SBOM are wired in `.github/workflows/ci.yml`), sign (for example with Cosign), and pin images by immutable `@sha256:...` digest before deployment.
5. **Upstream inference isolation:** Deploy GPU inference servers (CUDA or ROCm) in their own Deployment/namespace with GPU limits, model artifact controls, and TLS/mTLS.

## Observability

- **Structured logs:** Single-line JSON by default (`timestamp`, `level`, `logger`, `message`, `request_id`, `method`, `path`, `status_code`, `latency_ms`, `backend`, `outcome`). Prompts, chat messages, completions, authorization headers, and API keys are never logged.
- **OpenTelemetry tracing (optional):** Install `pip install "project-nexus[otel]"` and set `NEXUS_TRACING_ENABLED=true`. Spans record only allowlisted metadata (`nexus.request_id`, `nexus.backend`, `nexus.outcome`, `http.response.status_code`, normalized `url.path`) and propagate W3C `traceparent` headers.
- **Prometheus metrics:**
  - `nexus_requests_total{endpoint,backend,status_class,outcome}`
  - `nexus_request_duration_seconds{endpoint,backend,outcome}`
  - `nexus_backend_requests_total{backend,outcome}`
  - `nexus_backend_request_duration_seconds{backend}`
  - `nexus_backend_healthy{backend}`
  - `nexus_backend_circuit_open{backend}`
  - `nexus_in_flight_requests{endpoint}`
  - `nexus_active_streams{endpoint}`
  - `nexus_rate_limited_total{endpoint,reason}`
  - Arbitrary URL paths collapse into `endpoint="other"`, and request IDs, prompts, user IDs, and user-supplied model names are never used as metric labels.

## Troubleshooting & Known Limitations

- **`Configuration error: Exposing /metrics on the main server without authentication is disabled by default`**: Keep `metrics.mode: separate`, or set `NEXUS_METRICS_BEARER_TOKEN` when using `metrics.mode: main`.
- **`503 circuit_open` from `/v1/generate`, `/v1/chat/completions`, or `/readyz`**: The upstream backend failed `circuit_breaker.failure_threshold` consecutive times. Inspect upstream inference server health and logs; the breaker probes again after `cooldown_seconds`.
- **`429 rate_limited` or `429 concurrency_limited`**: Respect the `Retry-After` response header or tune `limits.rate_limit_requests_per_minute`, `limits.rate_limit_burst`, and `limits.max_concurrent_requests`.
- **`413 payload_too_large`**: Increase `limits.max_request_bytes` only after confirming downstream model context limits and memory budgets.
- **`nexus detect-gpu` shows CPU-only on a GPU machine**: Verify that `nvidia-smi` (NVIDIA) or `amd-smi` / `rocm-smi` (AMD) is on `PATH`, the host driver is loaded, and container GPU passthrough is enabled.
- **Known v0.1.0 limitations:** Single upstream backend per gateway instance (no multi-backend weighted routing/failover), per-replica in-memory rate limits, no real GPU hardware integration tests in CI, and no built-in OTLP exporter auto-configuration (configure standard OpenTelemetry SDK provider/env vars when enabling tracing).

## Contributing

1. Fork or branch from `main`, create a virtual environment, and install `pip install -e ".[dev]"`.
2. To add a new backend adapter, subclass `InferenceBackend` in `src/project_nexus/backends/`, register it in `backends/factory.py` and `config.py`, and add deterministic HTTP/mock unit tests that run without GPU hardware.
3. Ensure all local checks pass and keep `uv.lock` synchronized (`uv lock --check`) before opening a pull request:

```bash
uv lock --check
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=project_nexus
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
