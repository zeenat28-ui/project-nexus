"""Tests for optional NVIDIA CUDA and AMD ROCm GPU capability detection."""

from __future__ import annotations

import json
import subprocess

import pytest

from project_nexus.gpu import detect_gpu_capabilities


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["tool"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_detect_gpu_cpu_only_when_no_tools_found() -> None:
    report = detect_gpu_capabilities(
        which_fn=lambda _tool: None,
        path_exists_fn=lambda _path: False,
    )
    assert report.cpu_only is True
    assert report.detected_vendors == ()
    assert report.nvidia.available is False
    assert report.amd.available is False
    assert any("nvidia-smi" in msg for msg in report.nvidia.diagnostics)
    assert any("amd-smi" in msg for msg in report.amd.diagnostics)


def test_detect_gpu_nvidia_present() -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        assert argv[0] == "/usr/bin/nvidia-smi"
        return _completed(
            stdout=(
                "0, NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
                "1, NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
            )
        )

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda _p: False,
    )
    assert report.cpu_only is False
    assert report.detected_vendors == ("nvidia",)
    assert report.nvidia.available is True
    assert report.nvidia.driver_version == "550.54.15"
    assert len(report.nvidia.devices) == 2
    assert report.nvidia.devices[0].name == "NVIDIA A100-SXM4-80GB"
    assert report.nvidia.devices[0].memory_total_mib == 81920
    assert report.amd.available is False


def test_detect_gpu_amd_via_amd_smi() -> None:
    def fake_which(name: str) -> str | None:
        return "/opt/rocm/bin/amd-smi" if name == "amd-smi" else None

    amd_payload = [
        {
            "gpu": 0,
            "asic": {"market_name": "AMD Instinct MI300X"},
            "vram": {"size": {"value": 196608, "unit": "MB"}},
            "driver": {"version": "6.3.1"},
        }
    ]

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        assert argv[0] == "/opt/rocm/bin/amd-smi"
        return _completed(stdout=json.dumps(amd_payload))

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda _p: True,
    )
    assert report.cpu_only is False
    assert report.detected_vendors == ("amd",)
    assert report.amd.available is True
    assert report.amd.driver_version == "6.3.1"
    assert len(report.amd.devices) == 1
    assert report.amd.devices[0].name == "AMD Instinct MI300X"
    assert report.amd.devices[0].memory_total_mib == 196608


def test_detect_gpu_amd_fallback_to_rocm_smi() -> None:
    def fake_which(name: str) -> str | None:
        return "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None

    rocm_payload = {
        "card0": {
            "Card series": "AMD Instinct MI250X",
            "VRAM Total Memory (B)": str(64 * 1024 * 1024 * 1024),
        }
    }

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        assert argv[0] == "/opt/rocm/bin/rocm-smi"
        return _completed(stdout=json.dumps(rocm_payload))

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda _p: True,
    )
    assert report.amd.available is True
    assert report.amd.devices[0].name == "AMD Instinct MI250X"
    assert report.amd.devices[0].memory_total_mib == 65536


def test_detect_gpu_never_claims_available_on_failure_or_rocm_dir_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")

    def fake_which(name: str) -> str | None:
        if name == "nvidia-smi":
            return "/usr/bin/nvidia-smi"
        return None

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del argv, timeout
        return _completed(
            returncode=9,
            stderr="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
        )

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda p: p == "/opt/rocm",
    )
    assert report.cpu_only is True
    assert report.nvidia.available is False
    assert "exited with code 9" in report.nvidia.diagnostics[0]
    assert report.amd.available is False
    assert any("reporting AMD ROCm as unavailable" in msg for msg in report.amd.diagnostics)


def test_detect_gpu_handles_timeout_and_malformed_output() -> None:
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in {"nvidia-smi", "amd-smi"} else None

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        if "nvidia-smi" in argv[0]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
        return _completed(stdout="not valid json")

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda _p: False,
    )
    assert report.cpu_only is True
    assert report.nvidia.available is False
    assert any("timed out" in d for d in report.nvidia.diagnostics)
    assert report.amd.available is False
    assert any("non-JSON" in d for d in report.amd.diagnostics)


def test_detect_gpu_warns_on_mixed_vendor_host() -> None:
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in {"nvidia-smi", "amd-smi"} else None

    def fake_run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        if "nvidia-smi" in argv[0]:
            return _completed(stdout="0, NVIDIA L40S, 46068, 550.54.15\n")
        return _completed(
            stdout=json.dumps([{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X"}}])
        )

    report = detect_gpu_capabilities(
        which_fn=fake_which,
        run_fn=fake_run,
        path_exists_fn=lambda _p: True,
    )
    assert report.detected_vendors == ("nvidia", "amd")
    assert any("does NOT support mixed-vendor" in note for note in report.notes)
