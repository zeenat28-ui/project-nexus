"""Validated configuration loading for Project Nexus.

Precedence (lowest to highest):
1. Built-in safe defaults
2. YAML or TOML configuration file (via CLI ``--config`` or ``NEXUS_CONFIG_FILE``)
3. Environment variables prefixed with ``NEXUS_``
4. Explicit runtime/CLI overrides passed to ``load_settings``
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

AllowedJWTAlgorithm = Literal["RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA"]
AcceleratorType = Literal["unknown", "cpu", "nvidia_cuda", "amd_rocm"]
RoutingStrategy = Literal["priority", "weighted", "least_inflight", "latency_ewma", "cost_aware"]
_POOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or fails validation."""


class ServerSettings(BaseModel):
    """HTTP/HTTPS listener, inbound mTLS, and graceful shutdown settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    shutdown_drain_seconds: float = Field(default=10.0, ge=0.0, le=300.0)
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_ca_bundle_path: str | None = None
    tls_require_client_cert: bool = False

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_path and self.tls_key_path)

    @model_validator(mode="after")
    def validate_tls(self) -> ServerSettings:
        has_cert = self.tls_cert_path is not None
        has_key = self.tls_key_path is not None
        if has_cert != has_key:
            raise ValueError(
                "server.tls_cert_path and server.tls_key_path must both be set together "
                "to enable built-in TLS termination"
            )
        if self.tls_cert_path is not None and not self.tls_cert_path.strip():
            raise ValueError("server.tls_cert_path cannot be empty")
        if self.tls_key_path is not None and not self.tls_key_path.strip():
            raise ValueError("server.tls_key_path cannot be empty")
        if self.tls_ca_bundle_path is not None and not self.tls_ca_bundle_path.strip():
            raise ValueError("server.tls_ca_bundle_path cannot be empty")
        if self.tls_require_client_cert:
            if not self.tls_enabled:
                raise ValueError(
                    "server.tls_require_client_cert=true requires server.tls_cert_path and "
                    "server.tls_key_path"
                )
            if not self.tls_ca_bundle_path:
                raise ValueError(
                    "server.tls_require_client_cert=true requires server.tls_ca_bundle_path "
                    "to verify client certificates"
                )
        return self


class MockBackendSettings(BaseModel):
    """Deterministic settings for the built-in local mock backend."""

    model_config = ConfigDict(extra="forbid")

    latency_ms: float = Field(default=0.0, ge=0.0, le=30000.0)
    mode: Literal["normal", "timeout", "unavailable", "error", "unhealthy"] = "normal"
    response_prefix: str = Field(default="[mock]", max_length=64)


class BackendSettings(BaseModel):
    """Inference backend configuration including accelerator metadata, telemetry, and mTLS."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mock", "openai_compatible"] = "mock"
    model: str | None = Field(default=None, min_length=1, max_length=256)
    base_url: str | None = None
    api_style: Literal["chat_completions", "completions"] = "chat_completions"
    health_path: str = Field(default="/health", min_length=1, max_length=256)
    metrics_path: str | None = Field(default=None, min_length=1, max_length=256)
    max_kv_cache_usage_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    accelerator: AcceleratorType = "unknown"
    gpu_count: int = Field(default=0, ge=0, le=1024)
    weight: int = Field(default=1, ge=1, le=1000)
    cost_tier: int = Field(default=1, ge=1, le=100)
    api_key: SecretStr | None = None
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    read_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    verify_tls: bool = True
    ca_bundle_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    max_connections: int = Field(default=100, ge=1, le=1000)
    max_keepalive_connections: int = Field(default=20, ge=0, le=500)
    mock: MockBackendSettings = Field(default_factory=MockBackendSettings)

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> BackendSettings:
        if not self.health_path.startswith("/") or ".." in self.health_path:
            raise ValueError(
                "backend.health_path must be a relative URL path starting with '/' without '..'"
            )
        if self.metrics_path is not None and (
            not self.metrics_path.startswith("/") or ".." in self.metrics_path
        ):
            raise ValueError(
                "backend.metrics_path must be a relative URL path starting with '/' without '..'"
            )

        has_cert = self.client_cert_path is not None
        has_key = self.client_key_path is not None
        if has_cert != has_key:
            raise ValueError(
                "backend.client_cert_path and backend.client_key_path must both be set together "
                "for upstream mTLS"
            )
        if self.client_cert_path is not None and not self.client_cert_path.strip():
            raise ValueError("backend.client_cert_path cannot be empty")
        if self.client_key_path is not None and not self.client_key_path.strip():
            raise ValueError("backend.client_key_path cannot be empty")

        if self.type == "mock":
            if self.model is None:
                self.model = "mock-llm-v1"
            if self.accelerator == "unknown":
                self.accelerator = "cpu"
            return self

        if not self.base_url:
            raise ValueError(
                "backend.base_url (or NEXUS_BACKEND_BASE_URL) is required when "
                "backend.type='openai_compatible'"
            )
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "backend.base_url must be a valid http:// or https:// URL including a host"
            )
        self.base_url = self.base_url.rstrip("/")

        if not self.model:
            raise ValueError(
                "backend.model (or NEXUS_BACKEND_MODEL) is required when "
                "backend.type='openai_compatible'"
            )

        if self.ca_bundle_path is not None and not self.ca_bundle_path.strip():
            raise ValueError("backend.ca_bundle_path cannot be an empty string")

        return self


class RoutingSettings(BaseModel):
    """Multi-backend heterogeneous routing, scheduling strategy, and failover policy."""

    model_config = ConfigDict(extra="forbid")

    default_backend: str | None = None
    model_routes: dict[str, str | list[str]] = Field(default_factory=dict)
    strategy: RoutingStrategy = "priority"
    enable_failover: bool = False
    fallback_order: list[str] = Field(default_factory=list)


class JWTAuthSettings(BaseModel):
    """Asymmetric OIDC/JWT verification settings."""

    model_config = ConfigDict(extra="forbid")

    jwks_url: str | None = None
    public_key_pem: str | None = None
    issuer: str | None = Field(default=None, min_length=1)
    audience: str | None = Field(default=None, min_length=1)
    algorithms: tuple[AllowedJWTAlgorithm, ...] = ("RS256", "ES256")
    leeway_seconds: int = Field(default=30, ge=0, le=300)
    jwks_cache_seconds: int = Field(default=300, ge=10, le=86400)
    jwks_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)

    @model_validator(mode="after")
    def validate_jwt_settings(self) -> JWTAuthSettings:
        if not self.jwks_url and not self.public_key_pem:
            raise ValueError("auth.jwt requires either jwks_url or public_key_pem")
        if self.jwks_url is not None:
            parsed = urlparse(self.jwks_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("auth.jwt.jwks_url must be a valid http:// or https:// URL")
        if not self.algorithms:
            raise ValueError("auth.jwt.algorithms must contain at least one asymmetric algorithm")
        return self


class AuthSettings(BaseModel):
    """Authentication configuration supporting named/rotatable API keys or asymmetric JWTs."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "none", "api_key", "jwt"] = "auto"
    api_key: SecretStr | None = None
    api_keys: list[SecretStr] = Field(default_factory=list)
    named_api_keys: dict[str, SecretStr] = Field(default_factory=dict)
    jwt: JWTAuthSettings | None = None

    def all_api_keys(self) -> list[SecretStr]:
        keys: list[SecretStr] = []
        if self.api_key is not None:
            keys.append(self.api_key)
        keys.extend(self.api_keys)
        keys.extend(self.named_api_keys.values())
        return keys

    def resolved_mode(self) -> Literal["none", "api_key", "jwt"]:
        if self.mode != "auto":
            return self.mode
        if self.all_api_keys():
            return "api_key"
        if self.jwt is not None:
            return "jwt"
        return "none"

    @model_validator(mode="after")
    def validate_auth_mode(self) -> AuthSettings:
        for tenant_name in self.named_api_keys:
            if not _POOL_NAME_PATTERN.fullmatch(tenant_name):
                raise ValueError(
                    f"Invalid tenant name {tenant_name!r} in auth.named_api_keys; "
                    "use 1-64 characters from A-Z, a-z, 0-9, '.', '_', '-'."
                )
        if self.mode == "api_key" and not self.all_api_keys():
            raise ValueError(
                "auth.mode='api_key' requires auth.api_key, auth.api_keys, or "
                "auth.named_api_keys (NEXUS_AUTH_API_KEY / NEXUS_AUTH_API_KEYS)"
            )
        if self.mode == "jwt" and self.jwt is None:
            raise ValueError("auth.mode='jwt' requires an auth.jwt section")
        return self


class ModelPricingSettings(BaseModel):
    """Per-1K token pricing for a model in USD."""

    model_config = ConfigDict(extra="forbid")

    prompt_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)
    completion_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)


class TenantQuotaSettings(BaseModel):
    """Per-tenant token and USD cost caps within the metering window."""

    model_config = ConfigDict(extra="forbid")

    max_tokens_per_window: int | None = Field(default=None, ge=1)
    max_cost_usd_per_window: float | None = Field(default=None, gt=0.0)


class MeteringSettings(BaseModel):
    """Multi-tenant usage metering, cost accounting, and budget enforcement settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    window_seconds: int = Field(default=86_400, ge=60, le=2_592_000)
    sqlite_path: str | None = None
    default_prompt_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)
    default_completion_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)
    model_pricing: dict[str, ModelPricingSettings] = Field(default_factory=dict)
    default_max_tokens_per_window: int | None = Field(default=None, ge=1)
    default_max_cost_usd_per_window: float | None = Field(default=None, gt=0.0)
    tenants: dict[str, TenantQuotaSettings] = Field(default_factory=dict)


class LimitsSettings(BaseModel):
    """Request size, prompt length, token, timeout, concurrency, and rate-limit guardrails."""

    model_config = ConfigDict(extra="forbid")

    max_request_bytes: int = Field(default=65536, ge=256, le=10_485_760)
    max_prompt_chars: int = Field(default=16384, ge=1, le=500_000)
    max_tokens_limit: int = Field(default=2048, ge=1, le=32768)
    request_timeout_seconds: float = Field(default=35.0, gt=0.0, le=600.0)
    health_check_timeout_seconds: float = Field(default=3.0, gt=0.0, le=60.0)
    max_concurrent_requests: int = Field(default=64, ge=1, le=10_000)
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    rate_limit_requests_per_minute: int = Field(default=0, ge=0, le=1_000_000)
    rate_limit_burst: int | None = Field(default=None, ge=1, le=100_000)
    max_tracked_callers: int = Field(default=10_000, ge=10, le=1_000_000)
    rate_limit_redis_url: SecretStr | None = None
    rate_limit_redis_fail_open: bool = True
    rate_limit_redis_timeout_seconds: float = Field(default=1.0, gt=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_rate_limit_backend(self) -> LimitsSettings:
        if self.rate_limit_backend == "redis" and self.rate_limit_requests_per_minute > 0:
            if self.rate_limit_redis_url is None:
                raise ValueError(
                    "limits.rate_limit_redis_url (or NEXUS_LIMITS_RATE_LIMIT_REDIS_URL) is "
                    "required when limits.rate_limit_backend='redis' and rate limiting is enabled"
                )
            parsed = urlparse(self.rate_limit_redis_url.get_secret_value())
            if parsed.scheme not in {"redis", "rediss"} or not parsed.netloc:
                raise ValueError(
                    "limits.rate_limit_redis_url must be a valid redis:// or rediss:// URL"
                )
        return self


class CircuitBreakerSettings(BaseModel):
    """Failure isolation thresholds for upstream backend requests."""

    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(default=3, ge=1, le=100)
    cooldown_seconds: float = Field(default=15.0, gt=0.0, le=3600.0)


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


class TracingSettings(BaseModel):
    """Optional OpenTelemetry tracing and OTLP export settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    service_name: str = Field(default="project-nexus", min_length=1, max_length=128)
    otlp_endpoint: str | None = None
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    otlp_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)

    @model_validator(mode="after")
    def validate_otlp_endpoint(self) -> TracingSettings:
        if self.otlp_endpoint is not None:
            parsed = urlparse(self.otlp_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("tracing.otlp_endpoint must be a valid http:// or https:// URL")
        return self


class MetricsSettings(BaseModel):
    """Prometheus metrics & internal admin listener settings (secure by default)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["separate", "main", "disabled"] = "separate"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=9090, ge=1, le=65535)
    bearer_token: SecretStr | None = None
    allow_unauthenticated_main: bool = False
    tls_cert_path: str | None = None
    tls_key_path: str | None = None

    @model_validator(mode="after")
    def validate_metrics_security(self) -> MetricsSettings:
        if (self.tls_cert_path is None) != (self.tls_key_path is None):
            raise ValueError(
                "metrics.tls_cert_path and metrics.tls_key_path must both be set together"
            )
        if (
            self.mode == "main"
            and self.bearer_token is None
            and not self.allow_unauthenticated_main
        ):
            raise ValueError(
                "Exposing /metrics on the main server without authentication is disabled by "
                "default. Use metrics.mode='separate' (loopback/internal port), set "
                "metrics.bearer_token (NEXUS_METRICS_BEARER_TOKEN), or explicitly opt in with "
                "metrics.allow_unauthenticated_main=true."
            )
        return self


class NexusSettings(BaseModel):
    """Top-level validated configuration for the Project Nexus gateway."""

    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    backends: dict[str, BackendSettings] = Field(default_factory=dict)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    metering: MeteringSettings = Field(default_factory=MeteringSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

    @model_validator(mode="after")
    def validate_cross_section_rules(self) -> NexusSettings:
        if (
            self.metrics.mode == "separate"
            and self.metrics.port == self.server.port
            and self.metrics.host == self.server.host
        ):
            raise ValueError(
                "metrics.port cannot equal server.port when bound to the same host in "
                "metrics.mode='separate'"
            )

        if self.backends:
            for pool_name in self.backends:
                if not _POOL_NAME_PATTERN.fullmatch(pool_name):
                    raise ValueError(
                        f"Invalid backend pool name {pool_name!r}; use 1-64 characters from "
                        "A-Z, a-z, 0-9, '.', '_', '-'."
                    )

            if self.routing.default_backend is None:
                self.routing.default_backend = next(iter(self.backends))
            elif self.routing.default_backend not in self.backends:
                raise ValueError(
                    f"routing.default_backend={self.routing.default_backend!r} is not defined "
                    "under backends."
                )

            for model_id, target in self.routing.model_routes.items():
                pool_list = [target] if isinstance(target, str) else target
                if not pool_list:
                    raise ValueError(
                        f"routing.model_routes[{model_id!r}] must specify at least one pool."
                    )
                for target_pool in pool_list:
                    if target_pool not in self.backends:
                        raise ValueError(
                            f"routing.model_routes[{model_id!r}] references unknown backend pool "
                            f"{target_pool!r}."
                        )

            for fallback_pool in self.routing.fallback_order:
                if fallback_pool not in self.backends:
                    raise ValueError(
                        f"routing.fallback_order references unknown backend pool {fallback_pool!r}."
                    )
        else:
            if self.routing.model_routes or self.routing.fallback_order:
                raise ValueError(
                    "routing.model_routes and routing.fallback_order require named pools "
                    "configured under backends."
                )
            if self.routing.default_backend is not None:
                raise ValueError(
                    "routing.default_backend requires named pools configured under backends."
                )

        return self

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration view with secrets redacted."""
        raw = self.model_dump(mode="python")
        return {k: _redact_secrets(v) for k, v in raw.items()}


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {k: _redact_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item) for item in value]
    return value


def _parse_bool(raw: str, env_var: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean value for {env_var}: {raw!r}. Expected true/false or 1/0.")


def _parse_int(raw: str, env_var: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Invalid integer value for {env_var}: {raw!r}.") from exc


def _parse_float(raw: str, env_var: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Invalid numeric value for {env_var}: {raw!r}.") from exc


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value


def _load_config_file(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")

    suffix = config_path.suffix.lower()
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration file {config_path}: {exc}") from exc

    if suffix in {".yaml", ".yml"}:
        try:
            loaded = yaml.safe_load(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    elif suffix == ".toml":
        try:
            loaded = tomllib.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    else:
        raise ConfigError(
            f"Unsupported configuration file extension {suffix!r} for {config_path}. "
            "Use .yaml, .yml, or .toml."
        )

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        type_name = type(loaded).__name__
        raise ConfigError(
            f"Configuration root in {config_path} must be a mapping/table, got {type_name}."
        )
    return dict(loaded)


_ENV_MAPPINGS: tuple[
    tuple[str, tuple[str, ...], Literal["str", "int", "float", "bool", "upper", "csv"]], ...
] = (
    ("NEXUS_SERVER_HOST", ("server", "host"), "str"),
    ("NEXUS_SERVER_PORT", ("server", "port"), "int"),
    ("NEXUS_SERVER_SHUTDOWN_DRAIN_SECONDS", ("server", "shutdown_drain_seconds"), "float"),
    ("NEXUS_SERVER_TLS_CERT_PATH", ("server", "tls_cert_path"), "str"),
    ("NEXUS_SERVER_TLS_KEY_PATH", ("server", "tls_key_path"), "str"),
    ("NEXUS_SERVER_TLS_CA_BUNDLE_PATH", ("server", "tls_ca_bundle_path"), "str"),
    ("NEXUS_SERVER_TLS_REQUIRE_CLIENT_CERT", ("server", "tls_require_client_cert"), "bool"),
    ("NEXUS_BACKEND_TYPE", ("backend", "type"), "str"),
    ("NEXUS_BACKEND_MODEL", ("backend", "model"), "str"),
    ("NEXUS_BACKEND_BASE_URL", ("backend", "base_url"), "str"),
    ("NEXUS_BACKEND_API_STYLE", ("backend", "api_style"), "str"),
    ("NEXUS_BACKEND_HEALTH_PATH", ("backend", "health_path"), "str"),
    ("NEXUS_BACKEND_METRICS_PATH", ("backend", "metrics_path"), "str"),
    ("NEXUS_BACKEND_ACCELERATOR", ("backend", "accelerator"), "str"),
    ("NEXUS_BACKEND_API_KEY", ("backend", "api_key"), "str"),
    ("NEXUS_BACKEND_CONNECT_TIMEOUT_SECONDS", ("backend", "connect_timeout_seconds"), "float"),
    ("NEXUS_BACKEND_READ_TIMEOUT_SECONDS", ("backend", "read_timeout_seconds"), "float"),
    ("NEXUS_BACKEND_VERIFY_TLS", ("backend", "verify_tls"), "bool"),
    ("NEXUS_BACKEND_CA_BUNDLE_PATH", ("backend", "ca_bundle_path"), "str"),
    ("NEXUS_BACKEND_CLIENT_CERT_PATH", ("backend", "client_cert_path"), "str"),
    ("NEXUS_BACKEND_CLIENT_KEY_PATH", ("backend", "client_key_path"), "str"),
    ("NEXUS_BACKEND_MAX_CONNECTIONS", ("backend", "max_connections"), "int"),
    ("NEXUS_BACKEND_MAX_KEEPALIVE_CONNECTIONS", ("backend", "max_keepalive_connections"), "int"),
    ("NEXUS_MOCK_LATENCY_MS", ("backend", "mock", "latency_ms"), "float"),
    ("NEXUS_MOCK_MODE", ("backend", "mock", "mode"), "str"),
    ("NEXUS_MOCK_RESPONSE_PREFIX", ("backend", "mock", "response_prefix"), "str"),
    ("NEXUS_ROUTING_DEFAULT_BACKEND", ("routing", "default_backend"), "str"),
    ("NEXUS_ROUTING_STRATEGY", ("routing", "strategy"), "str"),
    ("NEXUS_ROUTING_ENABLE_FAILOVER", ("routing", "enable_failover"), "bool"),
    ("NEXUS_AUTH_MODE", ("auth", "mode"), "str"),
    ("NEXUS_AUTH_API_KEY", ("auth", "api_key"), "str"),
    ("NEXUS_AUTH_API_KEYS", ("auth", "api_keys"), "csv"),
    ("NEXUS_AUTH_JWT_JWKS_URL", ("auth", "jwt", "jwks_url"), "str"),
    ("NEXUS_AUTH_JWT_PUBLIC_KEY_PEM", ("auth", "jwt", "public_key_pem"), "str"),
    ("NEXUS_AUTH_JWT_ISSUER", ("auth", "jwt", "issuer"), "str"),
    ("NEXUS_AUTH_JWT_AUDIENCE", ("auth", "jwt", "audience"), "str"),
    ("NEXUS_METERING_ENABLED", ("metering", "enabled"), "bool"),
    ("NEXUS_METERING_WINDOW_SECONDS", ("metering", "window_seconds"), "int"),
    ("NEXUS_METERING_SQLITE_PATH", ("metering", "sqlite_path"), "str"),
    (
        "NEXUS_METERING_PROMPT_COST_PER_1K_USD",
        ("metering", "default_prompt_cost_per_1k_usd"),
        "float",
    ),
    (
        "NEXUS_METERING_COMPLETION_COST_PER_1K_USD",
        ("metering", "default_completion_cost_per_1k_usd"),
        "float",
    ),
    ("NEXUS_METERING_DEFAULT_MAX_TOKENS", ("metering", "default_max_tokens_per_window"), "int"),
    (
        "NEXUS_METERING_DEFAULT_MAX_COST_USD",
        ("metering", "default_max_cost_usd_per_window"),
        "float",
    ),
    ("NEXUS_LIMITS_MAX_REQUEST_BYTES", ("limits", "max_request_bytes"), "int"),
    ("NEXUS_LIMITS_MAX_PROMPT_CHARS", ("limits", "max_prompt_chars"), "int"),
    ("NEXUS_LIMITS_MAX_TOKENS_LIMIT", ("limits", "max_tokens_limit"), "int"),
    ("NEXUS_LIMITS_REQUEST_TIMEOUT_SECONDS", ("limits", "request_timeout_seconds"), "float"),
    (
        "NEXUS_LIMITS_HEALTH_CHECK_TIMEOUT_SECONDS",
        ("limits", "health_check_timeout_seconds"),
        "float",
    ),
    ("NEXUS_LIMITS_MAX_CONCURRENT_REQUESTS", ("limits", "max_concurrent_requests"), "int"),
    ("NEXUS_LIMITS_RATE_LIMIT_BACKEND", ("limits", "rate_limit_backend"), "str"),
    ("NEXUS_LIMITS_RATE_LIMIT_RPM", ("limits", "rate_limit_requests_per_minute"), "int"),
    ("NEXUS_LIMITS_RATE_LIMIT_BURST", ("limits", "rate_limit_burst"), "int"),
    ("NEXUS_LIMITS_RATE_LIMIT_REDIS_URL", ("limits", "rate_limit_redis_url"), "str"),
    ("NEXUS_LIMITS_RATE_LIMIT_REDIS_FAIL_OPEN", ("limits", "rate_limit_redis_fail_open"), "bool"),
    ("NEXUS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", ("circuit_breaker", "failure_threshold"), "int"),
    ("NEXUS_CIRCUIT_BREAKER_COOLDOWN_SECONDS", ("circuit_breaker", "cooldown_seconds"), "float"),
    ("NEXUS_LOG_LEVEL", ("logging", "level"), "upper"),
    ("NEXUS_LOG_FORMAT", ("logging", "format"), "str"),
    ("NEXUS_TRACING_ENABLED", ("tracing", "enabled"), "bool"),
    ("NEXUS_TRACING_SERVICE_NAME", ("tracing", "service_name"), "str"),
    ("NEXUS_TRACING_OTLP_ENDPOINT", ("tracing", "otlp_endpoint"), "str"),
    ("NEXUS_TRACING_SAMPLE_RATIO", ("tracing", "sample_ratio"), "float"),
    ("NEXUS_METRICS_MODE", ("metrics", "mode"), "str"),
    ("NEXUS_METRICS_HOST", ("metrics", "host"), "str"),
    ("NEXUS_METRICS_PORT", ("metrics", "port"), "int"),
    ("NEXUS_METRICS_BEARER_TOKEN", ("metrics", "bearer_token"), "str"),
    ("NEXUS_METRICS_ALLOW_UNAUTHENTICATED_MAIN", ("metrics", "allow_unauthenticated_main"), "bool"),
    ("NEXUS_METRICS_TLS_CERT_PATH", ("metrics", "tls_cert_path"), "str"),
    ("NEXUS_METRICS_TLS_KEY_PATH", ("metrics", "tls_key_path"), "str"),
)

_EMPTY_TO_NONE_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("auth", "api_key"),
        ("backend", "api_key"),
        ("backend", "ca_bundle_path"),
        ("backend", "client_cert_path"),
        ("backend", "client_key_path"),
        ("backend", "metrics_path"),
        ("limits", "rate_limit_redis_url"),
        ("metering", "sqlite_path"),
        ("metrics", "bearer_token"),
        ("metrics", "tls_cert_path"),
        ("metrics", "tls_key_path"),
        ("server", "tls_cert_path"),
        ("server", "tls_key_path"),
        ("server", "tls_ca_bundle_path"),
        ("tracing", "otlp_endpoint"),
    }
)


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> None:
    for env_name, path, kind in _ENV_MAPPINGS:
        if env_name not in env:
            continue
        raw = env[env_name]
        if raw == "":
            if path in _EMPTY_TO_NONE_PATHS:
                _set_nested(data, path, None)
                continue
            if kind == "csv":
                _set_nested(data, path, [])
                continue
            raise ConfigError(f"Environment variable {env_name} cannot be empty.")

        parsed: Any
        if kind == "str":
            parsed = raw
        elif kind == "upper":
            parsed = raw.strip().upper()
        elif kind == "int":
            parsed = _parse_int(raw, env_name)
        elif kind == "float":
            parsed = _parse_float(raw, env_name)
        elif kind == "bool":
            parsed = _parse_bool(raw, env_name)
        elif kind == "csv":
            parsed = _parse_csv_list(raw)
        else:  # pragma: no cover
            parsed = raw

        _set_nested(data, path, parsed)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Configuration validation failed:"]
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "root"
        lines.append(f"  - {loc}: {error['msg']}")
    return "\n".join(lines)


def load_settings(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> NexusSettings:
    """Load and validate Project Nexus settings with documented precedence."""
    env_map = os.environ if env is None else env
    resolved_path: Path | None = None

    if config_path is not None:
        resolved_path = Path(config_path)
    elif env_map.get("NEXUS_CONFIG_FILE"):
        resolved_path = Path(env_map["NEXUS_CONFIG_FILE"])

    data: dict[str, Any] = {}
    if resolved_path is not None:
        data = _load_config_file(resolved_path)

    _apply_env_overrides(data, env_map)

    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return NexusSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
