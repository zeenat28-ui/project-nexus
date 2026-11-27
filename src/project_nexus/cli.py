"""Command-line interface for Project Nexus."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Iterator, Sequence
from typing import Any

import uvicorn

from project_nexus import __version__
from project_nexus.app import create_app, create_metrics_app
from project_nexus.backends import create_backend
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.gpu import HardwareDetectionReport, detect_gpu_capabilities
from project_nexus.metrics import NexusMetrics
from project_nexus.verify import verify_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Project Nexus: LLM serving integration and operations gateway.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the Nexus HTTP serving gateway.")
    serve_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")
    serve_parser.add_argument("--host", help="Override server bind host.")
    serve_parser.add_argument("--port", type=int, help="Override server bind port.")

    check_parser = subparsers.add_parser(
        "check-config",
        help="Validate configuration and print sanitized settings.",
    )
    check_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")

    probe_parser = subparsers.add_parser(
        "probe",
        help="Run a single readiness health check against the configured backend.",
    )
    probe_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")

    verify_parser = subparsers.add_parser(
        "verify-backend",
        help="Run health, non-streaming, and streaming verification checks against a backend.",
    )
    verify_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")
    verify_parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of non-streaming latency samples (default: 3).",
    )
    verify_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="max_tokens used during verification requests (default: 32).",
    )

    gpu_parser = subparsers.add_parser(
        "detect-gpu",
        help="Inspect optional local NVIDIA CUDA / AMD ROCm GPU availability.",
    )
    gpu_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON output.",
    )
    gpu_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-tool command timeout in seconds (default: 5.0).",
    )

    subparsers.add_parser("version", help="Print Project Nexus version.")
    return parser


def _format_gpu_report_text(report: HardwareDetectionReport) -> str:
    vendors_summary = (
        ", ".join(report.detected_vendors) if report.detected_vendors else "none (CPU-only)"
    )
    lines = [
        f"Platform: {report.platform}",
        f"Detected GPU vendors: {vendors_summary}",
    ]
    for vendor_result in (report.nvidia, report.amd):
        status_label = "available" if vendor_result.available else "unavailable"
        header = f"  - {vendor_result.vendor.upper()}: {status_label}"
        if vendor_result.driver_version:
            header += f" (driver {vendor_result.driver_version})"
        lines.append(header)
        for dev in vendor_result.devices:
            mem = (
                f"{dev.memory_total_mib} MiB" if dev.memory_total_mib is not None else "unknown MiB"
            )
            lines.append(f"      [{dev.index}] {dev.name} ({mem})")
        for diag in vendor_result.diagnostics:
            lines.append(f"      diagnostic: {diag}")
    for note in report.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


class _NoSignalServer(uvicorn.Server):
    """Secondary Uvicorn server that leaves OS signal handling to the primary server."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def _serve_async(settings: NexusSettings) -> None:
    metrics = NexusMetrics()
    gateway_app = create_app(settings, metrics=metrics)

    main_server = uvicorn.Server(
        uvicorn.Config(
            gateway_app,
            host=settings.server.host,
            port=settings.server.port,
            access_log=False,
            log_level=settings.logging.level.lower(),
            timeout_graceful_shutdown=int(settings.server.shutdown_drain_seconds) + 5,
        )
    )

    if settings.metrics.mode != "separate":
        await main_server.serve()
        return

    metrics_secret = (
        settings.metrics.bearer_token.get_secret_value()
        if settings.metrics.bearer_token is not None
        else None
    )
    metrics_app = create_metrics_app(metrics, bearer_token=metrics_secret)
    metrics_server = _NoSignalServer(
        uvicorn.Config(
            metrics_app,
            host=settings.metrics.host,
            port=settings.metrics.port,
            access_log=False,
            log_level="warning",
        )
    )

    main_task = asyncio.create_task(main_server.serve())
    metrics_task = asyncio.create_task(metrics_server.serve())
    done, pending = await asyncio.wait(
        {main_task, metrics_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    main_server.should_exit = True
    metrics_server.should_exit = True
    for task in pending:
        await task
    for task in done:
        exc = task.exception()
        if exc is not None:
            raise exc


async def _run_probe(settings: NexusSettings) -> dict[str, Any]:
    backend = create_backend(settings.backend)
    await backend.startup()
    try:
        health = await backend.health_check(timeout=settings.limits.health_check_timeout_seconds)
    finally:
        await backend.shutdown()
    return {
        "healthy": health.healthy,
        "backend": health.backend,
        "latency_ms": health.latency_ms,
        "detail": health.detail,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``nexus`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "version":
        print(f"project-nexus {__version__}")
        return 0

    if args.command == "detect-gpu":
        if args.timeout <= 0:
            print("Error: --timeout must be greater than 0.", file=sys.stderr)
            return 2
        report = detect_gpu_capabilities(timeout_seconds=args.timeout)
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(_format_gpu_report_text(report))
        return 0

    overrides: dict[str, Any] = {}
    if args.command == "serve":
        server_overrides: dict[str, Any] = {}
        if args.host is not None:
            server_overrides["host"] = args.host
        if args.port is not None:
            server_overrides["port"] = args.port
        if server_overrides:
            overrides["server"] = server_overrides

    try:
        settings = load_settings(args.config, overrides=overrides or None)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check-config":
        print(json.dumps(settings.to_sanitized_dict(), indent=2))
        return 0

    if args.command == "probe":
        result = asyncio.run(_run_probe(settings))
        print(json.dumps(result, indent=2))
        return 0 if result["healthy"] else 1

    if args.command == "verify-backend":
        if args.iterations < 1 or args.max_tokens < 1:
            print("Error: --iterations and --max-tokens must be >= 1.", file=sys.stderr)
            return 2
        vreport = asyncio.run(
            verify_backend(
                settings,
                iterations=args.iterations,
                max_tokens=args.max_tokens,
            )
        )
        print(json.dumps(vreport.to_dict(), indent=2))
        return 0 if vreport.overall_passed else 1

    if args.command == "serve":
        try:
            asyncio.run(_serve_async(settings))
        except KeyboardInterrupt:
            return 0
        return 0

    return 2
