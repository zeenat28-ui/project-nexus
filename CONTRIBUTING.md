# Contributing to Project Nexus

Contributions are welcome under the [Apache License, Version 2.0](LICENSE).

## Local Development Setup (CPU-Only)

No GPU, CUDA toolkit, or ROCm installation is required for development or running tests.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

## Required Quality Checks

Run all checks locally before opening a pull request:

```bash
uv lock --check
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=project_nexus --cov-report=term-missing
```

If you modify dependencies in `pyproject.toml`, regenerate the lockfile with `uv lock` and commit `uv.lock`.

If you modify Kubernetes manifests under `deploy/kubernetes/`, verify both Kustomize renders:

```bash
kubectl kustomize deploy/kubernetes/base >/dev/null
kubectl kustomize deploy/kubernetes/overlays/ha >/dev/null
```

## Adding a New Inference Backend Adapter

1. Subclass `InferenceBackend` in `src/project_nexus/backends/` implementing `name`, `default_model`, `generate()`, `generate_stream()`, and `health_check()`.
2. Register the adapter type in `src/project_nexus/config.py` and `src/project_nexus/backends/factory.py`.
3. Add deterministic CPU-only unit tests (for example using `httpx.MockTransport`) covering happy paths, SSE streaming, timeouts, connection failures, error status codes, and credential redaction.
4. Keep claims in documentation strictly aligned with what is verified by tests or recorded hardware runs.

## Engineering Guardrails

- Never log prompts, chat messages, completions, or credentials.
- Never add high-cardinality labels (request IDs, raw prompts, user IDs, arbitrary user model strings) to Prometheus metrics.
- Avoid shell command construction from user input and always use `yaml.safe_load`.
- Keep changes focused and well-typed (`mypy --strict`).
