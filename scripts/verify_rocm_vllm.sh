#!/usr/bin/env bash
# Repeatable AMD ROCm + vLLM verification script for Project Nexus.
#
# Intended for AMD Developer Cloud or on-prem AMD Instinct nodes (MI250X / MI300X)
# running ROCm 6.x with an OpenAI-compatible vLLM ROCm server.
#
# Usage:
#   # Option A: Point at an already-running ROCm vLLM server
#   VLLM_BASE_URL=http://127.0.0.1:8001 MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
#     bash scripts/verify_rocm_vllm.sh
#
#   # Option B: Let the script launch local vLLM (requires vLLM ROCm installed in current Python)
#   START_LOCAL_VLLM=1 MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
#     bash scripts/verify_rocm_vllm.sh

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
START_LOCAL_VLLM="${START_LOCAL_VLLM:-0}"
ITERATIONS="${ITERATIONS:-5}"
BENCH_REQUESTS="${BENCH_REQUESTS:-10}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-2}"

VLLM_PID=""
GW_PID=""
CFG_FILE="$(mktemp /tmp/nexus-rocm-XXXXXX.yaml)"
VLLM_LOG="/tmp/nexus-rocm-vllm.log"

cleanup() {
  if [[ -n "${GW_PID}" ]] && kill -0 "${GW_PID}" 2>/dev/null; then
    kill "${GW_PID}" 2>/dev/null || true
    wait "${GW_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  rm -f "${CFG_FILE}"
}
trap cleanup EXIT

echo "=== 1. Inspecting AMD ROCm GPU availability ==="
if ! command -v amd-smi >/dev/null 2>&1 && ! command -v rocm-smi >/dev/null 2>&1; then
  echo "ERROR: Neither amd-smi nor rocm-smi found in PATH. Run this script on an AMD ROCm host." >&2
  exit 1
fi
if [[ ! -e /dev/kfd ]]; then
  echo "WARNING: /dev/kfd not visible in this environment; GPU compute passthrough may be missing." >&2
fi

GPU_JSON="$(nexus detect-gpu --json)"
echo "${GPU_JSON}"
python3 - <<'PY' "${GPU_JSON}"
import json, sys
report = json.loads(sys.argv[1])
assert report["amd"]["available"] is True, "nexus detect-gpu did not detect an available AMD ROCm GPU"
PY

if [[ "${START_LOCAL_VLLM}" == "1" ]]; then
  echo "=== 2. Launching local vLLM (ROCm) on port ${VLLM_PORT} ==="
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --model "${MODEL_ID}" \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!

  for _ in $(seq 1 100); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -n 60 "${VLLM_LOG}" >&2
      echo "ERROR: Local vLLM ROCm process exited early." >&2
      exit 1
    fi
    if curl -fsS "${VLLM_BASE_URL}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 3
  done
fi

echo "=== 3. Checking upstream ROCm vLLM /health at ${VLLM_BASE_URL} ==="
curl -fsS "${VLLM_BASE_URL}/health" >/dev/null

cat >"${CFG_FILE}" <<EOF
server:
  host: "127.0.0.1"
  port: 8000
  shutdown_drain_seconds: 5.0

backend:
  type: "openai_compatible"
  base_url: "${VLLM_BASE_URL}"
  model: "${MODEL_ID}"
  api_style: "chat_completions"
  health_path: "/health"
  connect_timeout_seconds: 5.0
  read_timeout_seconds: 60.0

limits:
  request_timeout_seconds: 65.0
  health_check_timeout_seconds: 5.0
  max_concurrent_requests: 32
  rate_limit_requests_per_minute: 240
  rate_limit_burst: 40

metrics:
  mode: "separate"
  host: "127.0.0.1"
  port: 9090
EOF

echo "=== 4. Running nexus verify-backend ==="
VERIFY_JSON="$(nexus verify-backend --config "${CFG_FILE}" --iterations "${ITERATIONS}")"
echo "${VERIFY_JSON}"

echo "=== 5. Running nexus benchmark (streaming) ==="
BENCH_JSON="$(nexus benchmark --config "${CFG_FILE}" --requests "${BENCH_REQUESTS}" --concurrency "${BENCH_CONCURRENCY}" --stream)"
echo "${BENCH_JSON}"

echo "=== 6. Running authenticated live nexus serve E2E check ==="
GW_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
METRICS_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"

NEXUS_AUTH_API_KEY="${GW_TOKEN}" \
NEXUS_METRICS_BEARER_TOKEN="${METRICS_TOKEN}" \
  nexus serve --config "${CFG_FILE}" >/tmp/nexus-rocm-gateway.log 2>&1 &
GW_PID=$!

for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

curl -fsS http://127.0.0.1:8000/readyz >/dev/null
UNAUTH_CODE="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)"
[[ "${UNAUTH_CODE}" == "401" ]]

curl -fsS -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GW_TOKEN}" \
  -H "X-Request-ID: rocm-e2e-1" \
  -d '{"prompt":"Say ROCm verified in two words.","max_tokens":24,"temperature":0.0}' >/dev/null

PUBLIC_METRICS_CODE="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/metrics)"
[[ "${PUBLIC_METRICS_CODE}" == "404" ]]

curl -fsS http://127.0.0.1:9090/metrics \
  -H "Authorization: Bearer ${METRICS_TOKEN}" | grep -q "nexus_requests_total"

echo "=== COPY THIS ROCm VERIFICATION SUMMARY ==="
python3 - <<'PY' "${MODEL_ID}" "${GPU_JSON}" "${VERIFY_JSON}" "${BENCH_JSON}"
import json, sys

model_id = sys.argv[1]
gpu = json.loads(sys.argv[2])["amd"]
verify = json.loads(sys.argv[3])
bench = json.loads(sys.argv[4])

env = {"model": model_id}
for mod in ("torch", "vllm"):
    try:
        m = __import__(mod)
        env[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        if mod == "torch":
            env["hip_version"] = getattr(m.version, "hip", None)
    except Exception:
        pass

print(json.dumps({
    "environment": env,
    "gpu": gpu,
    "verify_backend": {
        "overall_passed": verify["overall_passed"],
        "checks": verify["checks"],
        "latency_summary": verify["latency_summary"],
    },
    "benchmark_stream": bench,
    "gateway_e2e_passed": True,
}, indent=2))
PY
