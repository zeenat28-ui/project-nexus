"""Optional, read-only GPU runtime capability detection for NVIDIA CUDA and AMD ROCm.

This module inspects host CLI tools (``nvidia-smi``, ``amd-smi``, ``rocm-smi``)
without ever raising on missing drivers/tools and without falsely reporting a
GPU backend as available when probes fail or produce unparseable output.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

WhichFn = Callable[[str], str | None]
RunFn = Callable[..., subprocess.CompletedProcess[str]]
PathExistsFn = Callable[[str], bool]


@dataclass(frozen=True)
class GPUDeviceInfo:
    """Normalized metadata for a single detected GPU."""

    index: int
    name: str
    vendor: str
    memory_total_mib: int | None = None


@dataclass(frozen=True)
class VendorProbeResult:
    """Diagnostic result for a single GPU vendor runtime."""

    vendor: str
    available: bool
    tool: str | None = None
    driver_version: str | None = None
    devices: tuple[GPUDeviceInfo, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareDetectionReport:
    """Host GPU capability summary returned by ``detect_gpu_capabilities``."""

    platform: str
    detected_vendors: tuple[str, ...]
    cpu_only: bool
    nvidia: VendorProbeResult
    amd: VendorProbeResult
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_run(
    argv: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv is fixed executable + static read-only flags
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )


def _default_path_exists(path: str) -> bool:
    return Path(path).exists()


def _parse_int_or_none(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        cleaned = raw.strip().replace(",", "")
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def _probe_nvidia(
    *,
    timeout_seconds: float,
    which_fn: WhichFn,
    run_fn: RunFn,
) -> VendorProbeResult:
    tool_path = which_fn("nvidia-smi")
    if not tool_path:
        return VendorProbeResult(
            vendor="nvidia",
            available=False,
            tool=None,
            diagnostics=("nvidia-smi not found in PATH; NVIDIA CUDA driver/runtime not detected.",),
        )

    argv = [
        tool_path,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = run_fn(argv, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return VendorProbeResult(
            vendor="nvidia",
            available=False,
            tool=tool_path,
            diagnostics=(
                f"nvidia-smi timed out after {timeout_seconds:.1f}s; check NVIDIA driver health.",
            ),
        )
    except OSError as exc:
        return VendorProbeResult(
            vendor="nvidia",
            available=False,
            tool=tool_path,
            diagnostics=(f"Failed to execute nvidia-smi: {exc}",),
        )

    if completed.returncode != 0:
        err_hint = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = err_hint[0][:200] if err_hint else "no stderr output"
        return VendorProbeResult(
            vendor="nvidia",
            available=False,
            tool=tool_path,
            diagnostics=(f"nvidia-smi exited with code {completed.returncode}: {detail}",),
        )

    devices: list[GPUDeviceInfo] = []
    driver_version: str | None = None
    parse_warnings: list[str] = []

    for line_num, raw_line in enumerate(completed.stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            parse_warnings.append(f"Unexpected nvidia-smi CSV format on line {line_num}.")
            continue
        idx = _parse_int_or_none(parts[0])
        if idx is None:
            parse_warnings.append(f"Non-numeric GPU index on nvidia-smi line {line_num}.")
            continue
        name = parts[1] or f"NVIDIA GPU {idx}"
        mem_mib = _parse_int_or_none(parts[2])
        if parts[3] and driver_version is None:
            driver_version = parts[3]
        devices.append(
            GPUDeviceInfo(index=idx, name=name, vendor="nvidia", memory_total_mib=mem_mib)
        )

    if not devices:
        diagnostics = tuple(
            parse_warnings
            or ["nvidia-smi succeeded but reported zero parseable NVIDIA GPU devices."]
        )
        return VendorProbeResult(
            vendor="nvidia",
            available=False,
            tool=tool_path,
            driver_version=driver_version,
            diagnostics=diagnostics,
        )

    return VendorProbeResult(
        vendor="nvidia",
        available=True,
        tool=tool_path,
        driver_version=driver_version,
        devices=tuple(devices),
        diagnostics=tuple(parse_warnings),
    )


def _parse_amd_smi_json(stdout: str) -> tuple[list[GPUDeviceInfo], str | None, list[str]]:
    warnings: list[str] = []
    try:
        parsed = json.loads(stdout)
    except ValueError:
        return [], None, ["amd-smi returned non-JSON output."]

    entries: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        entries = [item for item in parsed if isinstance(item, dict)]
    elif isinstance(parsed, dict):
        gpu_data = parsed.get("gpu_data")
        if isinstance(gpu_data, list):
            entries = [item for item in gpu_data if isinstance(item, dict)]
        else:
            entries = [parsed]

    devices: list[GPUDeviceInfo] = []
    driver_version: str | None = None

    for pos, entry in enumerate(entries):
        idx = _parse_int_or_none(entry.get("gpu"))
        if idx is None:
            idx = pos

        asic_raw = entry.get("asic")
        asic: dict[str, Any] = asic_raw if isinstance(asic_raw, dict) else {}
        vram_raw = entry.get("vram")
        vram: dict[str, Any] = vram_raw if isinstance(vram_raw, dict) else {}
        driver_raw = entry.get("driver")
        driver: dict[str, Any] = driver_raw if isinstance(driver_raw, dict) else {}

        name_raw = asic.get("market_name") or asic.get("name") or entry.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            continue

        mem_mib: int | None = None
        vram_size = vram.get("size")
        if isinstance(vram_size, dict):
            raw_val = _parse_int_or_none(vram_size.get("value"))
            unit = str(vram_size.get("unit", "MB")).upper()
            if raw_val is not None:
                if unit in {"B", "BYTES"}:
                    mem_mib = raw_val // (1024 * 1024)
                elif unit in {"GB", "GIB"}:
                    mem_mib = raw_val * 1024
                else:
                    mem_mib = raw_val
        elif vram_size is not None:
            mem_mib = _parse_int_or_none(vram_size)

        drv = driver.get("version")
        if isinstance(drv, str) and drv.strip() and driver_version is None:
            driver_version = drv.strip()

        devices.append(
            GPUDeviceInfo(
                index=idx,
                name=name_raw.strip(),
                vendor="amd",
                memory_total_mib=mem_mib,
            )
        )

    if not devices:
        warnings.append("amd-smi JSON contained no recognizable AMD GPU entries.")
    return devices, driver_version, warnings


def _parse_rocm_smi_json(stdout: str) -> tuple[list[GPUDeviceInfo], list[str]]:
    warnings: list[str] = []
    try:
        parsed = json.loads(stdout)
    except ValueError:
        return [], ["rocm-smi returned non-JSON output."]

    if not isinstance(parsed, dict):
        return [], ["rocm-smi returned unexpected JSON root type."]

    devices: list[GPUDeviceInfo] = []
    for pos, (card_key, card_info) in enumerate(sorted(parsed.items())):
        if not isinstance(card_info, dict) or not str(card_key).lower().startswith("card"):
            continue
        suffix = str(card_key)[4:]
        idx = _parse_int_or_none(suffix)
        if idx is None:
            idx = pos

        name = (
            card_info.get("Card series")
            or card_info.get("Card Series")
            or card_info.get("Card model")
            or card_info.get("Card Model")
            or f"AMD GPU {idx}"
        )
        vram_bytes = _parse_int_or_none(card_info.get("VRAM Total Memory (B)"))
        mem_mib = (vram_bytes // (1024 * 1024)) if vram_bytes is not None else None
        devices.append(
            GPUDeviceInfo(
                index=idx,
                name=str(name).strip(),
                vendor="amd",
                memory_total_mib=mem_mib,
            )
        )

    if not devices:
        warnings.append("rocm-smi JSON contained no recognizable card entries.")
    return devices, warnings


def _probe_amd(
    *,
    timeout_seconds: float,
    which_fn: WhichFn,
    run_fn: RunFn,
    path_exists_fn: PathExistsFn,
) -> VendorProbeResult:
    diagnostics: list[str] = []

    amd_smi_path = which_fn("amd-smi")
    if amd_smi_path:
        try:
            completed = run_fn([amd_smi_path, "static", "--json"], timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            diagnostics.append(f"amd-smi timed out after {timeout_seconds:.1f}s.")
        except OSError as exc:
            diagnostics.append(f"Failed to execute amd-smi: {exc}")
        else:
            if completed.returncode == 0:
                devices, driver_version, warnings = _parse_amd_smi_json(completed.stdout)
                diagnostics.extend(warnings)
                if devices:
                    return VendorProbeResult(
                        vendor="amd",
                        available=True,
                        tool=amd_smi_path,
                        driver_version=driver_version,
                        devices=tuple(devices),
                        diagnostics=tuple(diagnostics),
                    )
            else:
                err_hint = (completed.stderr or completed.stdout or "").strip().splitlines()
                detail = err_hint[0][:200] if err_hint else "no stderr output"
                diagnostics.append(f"amd-smi exited with code {completed.returncode}: {detail}")

    rocm_smi_path = which_fn("rocm-smi")
    if rocm_smi_path:
        try:
            completed = run_fn(
                [rocm_smi_path, "--showproductname", "--showmeminfo", "vram", "--json"],
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            diagnostics.append(f"rocm-smi timed out after {timeout_seconds:.1f}s.")
        except OSError as exc:
            diagnostics.append(f"Failed to execute rocm-smi: {exc}")
        else:
            if completed.returncode == 0:
                devices, warnings = _parse_rocm_smi_json(completed.stdout)
                diagnostics.extend(warnings)
                if devices:
                    return VendorProbeResult(
                        vendor="amd",
                        available=True,
                        tool=rocm_smi_path,
                        devices=tuple(devices),
                        diagnostics=tuple(diagnostics),
                    )
            else:
                err_hint = (completed.stderr or completed.stdout or "").strip().splitlines()
                detail = err_hint[0][:200] if err_hint else "no stderr output"
                diagnostics.append(f"rocm-smi exited with code {completed.returncode}: {detail}")

    if not amd_smi_path and not rocm_smi_path:
        diagnostics.append(
            "Neither amd-smi nor rocm-smi found in PATH; AMD ROCm runtime not detected."
        )

    rocm_env = os.environ.get("ROCM_PATH")
    if rocm_env or path_exists_fn("/opt/rocm"):
        diagnostics.append(
            "ROCm directory/environment variable is present, but no AMD GPU was verified via "
            "amd-smi or rocm-smi; reporting AMD ROCm as unavailable."
        )

    return VendorProbeResult(
        vendor="amd",
        available=False,
        tool=amd_smi_path or rocm_smi_path,
        diagnostics=tuple(diagnostics),
    )


def detect_gpu_capabilities(
    *,
    timeout_seconds: float = 5.0,
    which_fn: WhichFn = shutil.which,
    run_fn: RunFn = _default_run,
    path_exists_fn: PathExistsFn = _default_path_exists,
) -> HardwareDetectionReport:
    """Detect NVIDIA CUDA and AMD ROCm GPUs safely without crashing on CPU-only hosts."""
    nvidia = _probe_nvidia(
        timeout_seconds=timeout_seconds,
        which_fn=which_fn,
        run_fn=run_fn,
    )
    amd = _probe_amd(
        timeout_seconds=timeout_seconds,
        which_fn=which_fn,
        run_fn=run_fn,
        path_exists_fn=path_exists_fn,
    )

    detected: list[str] = []
    if nvidia.available:
        detected.append("nvidia")
    if amd.available:
        detected.append("amd")

    notes: list[str] = []
    if not detected:
        notes.append(
            "No verified GPU runtime detected on this host. Nexus gateway runs on CPU and "
            "can connect to remote GPU inference backends or the local mock backend."
        )
    if len(detected) > 1:
        notes.append(
            "Both NVIDIA and AMD GPUs were detected on this host. Project Nexus does NOT "
            "support mixed-vendor multi-GPU execution; run separate vendor-specific inference "
            "backends (e.g., dedicated CUDA or ROCm vLLM servers)."
        )

    return HardwareDetectionReport(
        platform=platform.system(),
        detected_vendors=tuple(detected),
        cpu_only=len(detected) == 0,
        nvidia=nvidia,
        amd=amd,
        notes=tuple(notes),
    )
