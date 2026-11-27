"""Tests for built-in TLS termination and inbound mTLS client-cert enforcement."""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import socket
import ssl
from pathlib import Path

import httpx
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from project_nexus.app import create_app
from project_nexus.cli import build_uvicorn_tls_kwargs
from project_nexus.config import ConfigError, NexusSettings, load_settings


def _write_key(path: Path, key: rsa.RSAPrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _generate_pki(tmp_path: Path) -> dict[str, Path]:
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Nexus Test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")), x509.DNSName("localhost")]
            ),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    cli_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cli_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nexus-client")]))
        .issuer_name(ca_name)
        .public_key(cli_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    paths = {
        "ca_cert": tmp_path / "ca.crt",
        "server_cert": tmp_path / "server.crt",
        "server_key": tmp_path / "server.key",
        "client_cert": tmp_path / "client.crt",
        "client_key": tmp_path / "client.key",
    }
    _write_cert(paths["ca_cert"], ca_cert)
    _write_cert(paths["server_cert"], srv_cert)
    _write_key(paths["server_key"], srv_key)
    _write_cert(paths["client_cert"], cli_cert)
    _write_key(paths["client_key"], cli_key)
    return paths


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_tls_config_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="tls_cert_path"):
        load_settings(env={"NEXUS_SERVER_TLS_CERT_PATH": "/tmp/cert.pem"})

    with pytest.raises(ConfigError, match="tls_ca_bundle_path"):
        load_settings(
            env={
                "NEXUS_SERVER_TLS_CERT_PATH": "/tmp/cert.pem",
                "NEXUS_SERVER_TLS_KEY_PATH": "/tmp/key.pem",
                "NEXUS_SERVER_TLS_REQUIRE_CLIENT_CERT": "true",
            }
        )

    with pytest.raises(ConfigError, match="not found"):
        build_uvicorn_tls_kwargs(
            cert_path=str(tmp_path / "missing.crt"),
            key_path=str(tmp_path / "missing.key"),
        )


@pytest.mark.asyncio
async def test_live_https_and_inbound_mtls_enforcement(tmp_path: Path) -> None:
    pki = _generate_pki(tmp_path)
    port = _free_tcp_port()
    tls_kwargs = build_uvicorn_tls_kwargs(
        cert_path=str(pki["server_cert"]),
        key_path=str(pki["server_key"]),
        ca_bundle_path=str(pki["ca_cert"]),
        require_client_cert=True,
    )

    app = create_app(NexusSettings(), configure_loggers=False)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="error",
            **tls_kwargs,
        )
    )
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "HTTPS Uvicorn test server failed to start"

        # 1. Valid mTLS client succeeds
        mtls_ctx = ssl.create_default_context(cafile=str(pki["ca_cert"]))
        mtls_ctx.load_cert_chain(certfile=str(pki["client_cert"]), keyfile=str(pki["client_key"]))
        async with httpx.AsyncClient(verify=mtls_ctx, timeout=5.0) as client:
            health = await client.get(f"https://127.0.0.1:{port}/healthz")
            assert health.status_code == 200
            gen = await client.post(
                f"https://127.0.0.1:{port}/v1/generate",
                json={"prompt": "Hello over mTLS"},
            )
            assert gen.status_code == 200
            assert gen.json()["output"] == "[mock] Hello over mTLS"

        # 2. Client without client certificate is rejected at TLS handshake
        no_cert_ctx = ssl.create_default_context(cafile=str(pki["ca_cert"]))
        async with httpx.AsyncClient(verify=no_cert_ctx, timeout=5.0) as client:
            with pytest.raises(httpx.HTTPError):
                await client.get(f"https://127.0.0.1:{port}/healthz")
    finally:
        server.should_exit = True
        await server_task
