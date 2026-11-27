"""CLI and package invocation smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from project_nexus import __version__
from project_nexus.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version"])
    assert exit_code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_check_config_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NEXUS_AUTH_API_KEY", "cli-secret-token")
    exit_code = main(["check-config", "--config", str(REPO_ROOT / "config" / "nexus.mock.yaml")])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "cli-secret-token" not in output
    parsed = json.loads(output)
    assert parsed["auth"]["api_key"] == "***REDACTED***"
    assert parsed["backend"]["type"] == "mock"


def test_cli_check_config_invalid_returns_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("backend:\n  type: openai_compatible\n", encoding="utf-8")
    exit_code = main(["check-config", "--config", str(bad)])
    assert exit_code == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_probe_mock_healthy_and_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NEXUS_MOCK_MODE", "normal")
    ok_code = main(["probe"])
    assert ok_code == 0
    assert json.loads(capsys.readouterr().out)["healthy"] is True

    monkeypatch.setenv("NEXUS_MOCK_MODE", "unhealthy")
    fail_code = main(["probe"])
    assert fail_code == 1
    assert json.loads(capsys.readouterr().out)["healthy"] is False


def test_cli_detect_gpu_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    code_json = main(["detect-gpu", "--json"])
    assert code_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert "detected_vendors" in payload
    assert "nvidia" in payload
    assert "amd" in payload

    code_text = main(["detect-gpu"])
    assert code_text == 0
    assert "Detected GPU vendors:" in capsys.readouterr().out


def test_python_module_entrypoint_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "project_nexus", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert __version__ in completed.stdout
