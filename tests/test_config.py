"""Tests for configuration loading, precedence, validation, and secret redaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_nexus.config import ConfigError, NexusSettings, load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_settings_are_safe_and_cpu_only() -> None:
    settings = load_settings(env={})
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000
    assert settings.backend.type == "mock"
    assert settings.backend.model == "mock-llm-v1"
    assert settings.backend.verify_tls is True
    assert settings.metrics.mode == "separate"
    assert settings.metrics.host == "127.0.0.1"
    assert settings.metrics.port == 9090
    assert settings.metrics.allow_unauthenticated_main is False


def test_yaml_env_and_override_precedence(tmp_path: Path) -> None:
    config_file = tmp_path / "nexus.yaml"
    config_file.write_text(
        """
server:
  host: "0.0.0.0"
  port: 8100
backend:
  type: "mock"
  model: "from-file"
  mock:
    latency_ms: 12.5
logging:
  level: "WARNING"
""",
        encoding="utf-8",
    )

    settings = load_settings(
        config_file,
        env={
            "NEXUS_SERVER_PORT": "8200",
            "NEXUS_BACKEND_MODEL": "from-env",
            "NEXUS_LOG_LEVEL": "debug",
        },
        overrides={"server": {"port": 8300}},
    )

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 8300  # override > env > file
    assert settings.backend.model == "from-env"  # env > file
    assert settings.backend.mock.latency_ms == 12.5  # from file
    assert settings.logging.level == "DEBUG"  # env uppercased


def test_toml_config_file_loading(tmp_path: Path) -> None:
    config_file = tmp_path / "nexus.toml"
    config_file.write_text(
        """
[server]
host = "127.0.0.1"
port = 8400

[backend]
type = "openai_compatible"
base_url = "https://inference.internal.example:8443/"
model = "meta-llama/Llama-3.1-8B-Instruct"
api_style = "completions"
health_path = "/v1/models"
""",
        encoding="utf-8",
    )

    settings = load_settings(config_file, env={})
    assert settings.server.port == 8400
    assert settings.backend.type == "openai_compatible"
    assert settings.backend.base_url == "https://inference.internal.example:8443"
    assert settings.backend.api_style == "completions"
    assert settings.backend.health_path == "/v1/models"


def test_config_file_from_env_var(tmp_path: Path) -> None:
    config_file = tmp_path / "env.yaml"
    config_file.write_text("server:\n  port: 8555\n", encoding="utf-8")

    settings = load_settings(env={"NEXUS_CONFIG_FILE": str(config_file)})
    assert settings.server.port == 8555


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("server:\n  unknown_field: 123\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown_field"):
        load_settings(config_file, env={})


def test_unsafe_yaml_tags_are_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "unsafe.yaml"
    config_file.write_text(
        "server:\n  host: !!python/object/apply:os.system ['echo pwn']\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_settings(config_file, env={})


def test_openai_compatible_requires_base_url_and_model() -> None:
    with pytest.raises(ConfigError, match="base_url"):
        load_settings(env={"NEXUS_BACKEND_TYPE": "openai_compatible", "NEXUS_BACKEND_MODEL": "m"})

    with pytest.raises(ConfigError, match="http:// or https://"):
        load_settings(
            env={
                "NEXUS_BACKEND_TYPE": "openai_compatible",
                "NEXUS_BACKEND_BASE_URL": "ftp://example.com",
                "NEXUS_BACKEND_MODEL": "m",
            }
        )

    with pytest.raises(ConfigError, match=r"backend\.model"):
        load_settings(
            env={
                "NEXUS_BACKEND_TYPE": "openai_compatible",
                "NEXUS_BACKEND_BASE_URL": "http://127.0.0.1:8001",
            }
        )


def test_metrics_on_main_app_requires_auth_or_explicit_opt_in() -> None:
    with pytest.raises(ConfigError, match=r"metrics\.bearer_token"):
        load_settings(env={"NEXUS_METRICS_MODE": "main"})

    with_token = load_settings(
        env={
            "NEXUS_METRICS_MODE": "main",
            "NEXUS_METRICS_BEARER_TOKEN": "super-secret-metrics-token",
        }
    )
    assert with_token.metrics.mode == "main"

    opted_in = load_settings(
        env={
            "NEXUS_METRICS_MODE": "main",
            "NEXUS_METRICS_ALLOW_UNAUTHENTICATED_MAIN": "true",
        }
    )
    assert opted_in.metrics.allow_unauthenticated_main is True


def test_separate_metrics_port_collision_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"cannot equal server\.port"):
        load_settings(
            env={
                "NEXUS_SERVER_HOST": "127.0.0.1",
                "NEXUS_SERVER_PORT": "8000",
                "NEXUS_METRICS_MODE": "separate",
                "NEXUS_METRICS_HOST": "127.0.0.1",
                "NEXUS_METRICS_PORT": "8000",
            }
        )


def test_invalid_env_var_types_raise_actionable_errors() -> None:
    with pytest.raises(ConfigError, match="NEXUS_SERVER_PORT"):
        load_settings(env={"NEXUS_SERVER_PORT": "not-an-int"})

    with pytest.raises(ConfigError, match="NEXUS_BACKEND_VERIFY_TLS"):
        load_settings(env={"NEXUS_BACKEND_VERIFY_TLS": "maybe"})

    with pytest.raises(ConfigError, match="NEXUS_MOCK_LATENCY_MS"):
        load_settings(env={"NEXUS_MOCK_LATENCY_MS": "fast"})


def test_missing_or_unsupported_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.yaml", env={})

    txt_file = tmp_path / "config.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported configuration file extension"):
        load_settings(txt_file, env={})


def test_secrets_are_redacted_in_sanitized_dict_and_repr() -> None:
    gateway_secret = "gw-secret-value-9876"
    backend_secret = "be-secret-value-5432"
    metrics_secret = "mt-secret-value-1098"

    settings = load_settings(
        env={
            "NEXUS_AUTH_API_KEY": gateway_secret,
            "NEXUS_BACKEND_API_KEY": backend_secret,
            "NEXUS_METRICS_BEARER_TOKEN": metrics_secret,
        }
    )

    sanitized = settings.to_sanitized_dict()
    assert sanitized["auth"]["api_key"] == "***REDACTED***"
    assert sanitized["backend"]["api_key"] == "***REDACTED***"
    assert sanitized["metrics"]["bearer_token"] == "***REDACTED***"

    serialized = f"{sanitized!r} {settings!r}"
    assert gateway_secret not in serialized
    assert backend_secret not in serialized
    assert metrics_secret not in serialized


def test_shipped_example_configs_validate() -> None:
    mock_cfg = load_settings(REPO_ROOT / "config" / "nexus.mock.yaml", env={})
    assert isinstance(mock_cfg, NexusSettings)
    assert mock_cfg.backend.type == "mock"

    openai_cfg = load_settings(REPO_ROOT / "config" / "nexus.openai-compat.example.yaml", env={})
    assert openai_cfg.backend.type == "openai_compatible"
    assert openai_cfg.backend.verify_tls is True
