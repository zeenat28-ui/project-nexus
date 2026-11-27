# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.2.x` | Yes |
| `0.1.x` | Security fixes via upgrade to `0.2.x` |

## Reporting a Vulnerability

Please do **not** open a public GitHub Issue for security vulnerabilities.

Report vulnerabilities privately through GitHub's **Security Advisories** feature:
`https://github.com/zeenat28-ui/project-nexus/security/advisories/new`

Include:
- Affected version or commit SHA
- Reproduction steps or proof-of-concept request/configuration
- Expected impact and any suggested remediation

## Threat Model & Deployment Trust Boundaries

Project Nexus is a stateless HTTP control/integration gateway placed in front of external LLM inference engines. Deployers are responsible for enforcing the following boundaries:

1. **Ingress TLS:** Nexus serves plaintext HTTP. Always terminate TLS at an Ingress controller, load balancer, API gateway, or mTLS service mesh before exposing Nexus outside `127.0.0.1`.
2. **Authentication:** Enable `auth.mode: api_key` (`NEXUS_AUTH_API_KEY` / `NEXUS_AUTH_API_KEYS`) or `auth.mode: jwt` (asymmetric `RS256`/`ES256` JWKS) on all non-localhost deployments. Never commit credentials to YAML/TOML files or Git.
3. **Metrics & Internal Endpoints:** Keep `metrics.mode: separate` (default) so `/metrics` is never mounted on the client-facing listener. Require `NEXUS_METRICS_BEARER_TOKEN` and Kubernetes `NetworkPolicy` isolation on port `9090`.
4. **Upstream Inference Isolation:** Enable HTTPS (`backend.verify_tls: true`, the default) and optional mTLS client certificates (`backend.client_cert_path` / `backend.client_key_path`) between Nexus and upstream inference servers. Outbound HTTP redirects are disabled (`follow_redirects=False`) to prevent credential leakage.
5. **Sensitive Payload Hygiene:** Nexus does not log prompts, chat messages, completions, or bearer tokens in structured logs, validation errors, Prometheus labels, or OpenTelemetry span attributes.
6. **Container & Kubernetes Hardening:** Run as non-root UID `10001`, keep `readOnlyRootFilesystem: true`, drop all Linux capabilities, pin images by immutable `@sha256:` digest, and provision secrets via External Secrets Operator, Vault, or KMS-backed stores.
