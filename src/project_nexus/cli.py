"""Command-line interface for Project Nexus."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import ssl
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import uvicorn

from project_nexus import __version__
from project_nexus.app import create_app, create_metrics_app
from project_nexus.backends import BackendRouter
from project_nexus.config import ConfigError, NexusSettings, load_settings
from project_nexus.gpu import HardwareDetectionReport, detect_gpu_capabilities
from project_nexus.metrics import NexusMetrics
from project_nexus.verify import benchmark_backend, verify_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Project Nexus: LLM serving integration and operations gateway.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the Nexus HTTP/HTTPS serving gateway.")
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
        help="Run a readiness health check against all configured backends.",
    )
    probe_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")

    verify_parser = subparsers.add_parser(
        "verify-backend",
        help="Run health, non-streaming, and streaming verification checks against a backend.",
    )
    verify_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")
    verify_parser.add_argument(
        "--backend",
        dest="backend_name",
        help="Optional named backend pool when using multi-backend config.",
    )
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

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run a bounded-concurrency latency and throughput benchmark against a backend.",
    )
    bench_parser.add_argument("--config", help="Path to YAML or TOML configuration file.")
    bench_parser.add_argument(
        "--backend",
        dest="backend_name",
        help="Optional named backend pool when using multi-backend config.",
    )
    bench_parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Total requests to dispatch (default: 10).",
    )
    bench_parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent in-flight requests (default: 2).",
    )
    bench_parser.add_argument(
        "--stream",
        action="store_true",
        help="Benchmark SSE streaming mode and report TTFT percentiles.",
    )
    bench_parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="max_tokens per benchmark request (default: 48).",
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


def build_uvicorn_tls_kwargs(
    *,
    cert_path: str | None,
    key_path: str | None,
    ca_bundle_path: str | None = None,
    require_client_cert: bool = False,
) -> dict[str, Any]:
    """Validate TLS file paths and return Uvicorn SSL configuration kwargs."""
    if not cert_path or not key_path:
        return {}
    for label, raw_path in (
        ("TLS certificate", cert_path),
        ("TLS private key", key_path),
    ):
        if not Path(raw_path).is_file():
            raise ConfigError(f"{label} file not found: {raw_path}")

    kwargs: dict[str, Any] = {
        "ssl_certfile": cert_path,
        "ssl_keyfile": key_path,
        "ssl_version": ssl.PROTOCOL_TLS_SERVER,
    }
    if ca_bundle_path:
        if not Path(ca_bundle_path).is_file():
            raise ConfigError(f"TLS CA bundle file not found: {ca_bundle_path}")
        kwargs["ssl_ca_certs"] = ca_bundle_path
        kwargs["ssl_cert_reqs"] = (
            int(ssl.CERT_REQUIRED) if require_client_cert else int(ssl.CERT_OPTIONAL)
        )
    return kwargs


async def _serve_async(settings: NexusSettings) -> None:
    main_tls = build_uvicorn_tls_kwargs(
        cert_path=settings.server.tls_cert_path,
        key_path=settings.server.tls_key_path,
        ca_bundle_path=settings.server.tls_ca_bundle_path,
        require_client_cert=settings.server.tls_require_client_cert,
    )
    metrics_tls = build_uvicorn_tls_kwargs(
        cert_path=settings.metrics.tls_cert_path,
        key_path=settings.metrics.tls_key_path,
    )

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
            **main_tls,
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
    metrics_app = create_metrics_app(
        metrics,
        bearer_token=metrics_secret,
        usage_meter=gateway_app.state.usage_meter,
        router=gateway_app.state.router,
    )
    metrics_server = _NoSignalServer(
        uvicorn.Config(
            metrics_app,
            host=settings.metrics.host,
            port=settings.metrics.port,
            access_log=False,
            log_level="warning",
            **metrics_tls,
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
    router = BackendRouter.from_settings(settings)
    await router.startup()
    try:
        any_healthy, status_map = await router.check_all_health(
            timeout_seconds=settings.limits.health_check_timeout_seconds
        )
    finally:
        await router.shutdown()

    if not router.is_multi_backend:
        sole = status_map[router.default_route]
        return {
            "healthy": bool(sole["healthy"]),
            "backend": sole["backend"],
            "latency_ms": sole.get("latency_ms"),
            "detail": sole["detail"],
        }
    return {
        "healthy": any_healthy,
        "default_backend": router.default_route,
        "backends": status_map,
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
        try:
            vreport = asyncio.run(
                verify_backend(
                    settings,
                    backend_name=args.backend_name,
                    iterations=args.iterations,
                    max_tokens=args.max_tokens,
                )
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(vreport.to_dict(), indent=2))
        return 0 if vreport.overall_passed else 1

    if args.command == "benchmark":
        if args.requests < 1 or args.concurrency < 1 or args.max_tokens < 1:
            print(
                "Error: --requests, --concurrency, and --max-tokens must be >= 1.",
                file=sys.stderr,
            )
            return 2
        try:
            breport = asyncio.run(
                benchmark_backend(
                    settings,
                    backend_name=args.backend_name,
                    total_requests=args.requests,
                    concurrency=args.concurrency,
                    stream=args.stream,
                    max_tokens=args.max_tokens,
                )
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(breport.to_dict(), indent=2))
        return 0 if breport.failed == 0 else 1

    if args.command == "serve":
        try:
            asyncio.run(_serve_async(settings))
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return 0
        return 0

    return 2
