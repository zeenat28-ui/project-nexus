"""Tests for rotatable API keys and asymmetric OIDC/JWT authentication."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

from project_nexus.app import create_app
from project_nexus.auth import GatewayAuthenticator
from project_nexus.config import ConfigError, load_settings


def _generate_rsa_pem_pair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pub_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return priv_pem, pub_pem


def _generate_ec_pem_pair() -> tuple[str, str]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pub_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return priv_pem, pub_pem


def test_rotatable_multi_api_keys_and_redaction() -> None:
    settings = load_settings(
        env={
            "NEXUS_AUTH_MODE": "api_key",
            "NEXUS_AUTH_API_KEY": "primary-key-v1",
            "NEXUS_AUTH_API_KEYS": "rotated-key-v2, rotated-key-v3",
        }
    )
    sanitized = settings.to_sanitized_dict()
    assert sanitized["auth"]["api_key"] == "***REDACTED***"
    assert sanitized["auth"]["api_keys"] == ["***REDACTED***", "***REDACTED***"]
    assert "rotated-key-v2" not in repr(settings)

    app = create_app(settings, configure_loggers=False)
    with TestClient(app) as client:
        assert client.post("/v1/generate", json={"prompt": "hi"}).status_code == 401
        for key in ("primary-key-v1", "rotated-key-v2", "rotated-key-v3"):
            resp = client.post(
                "/v1/generate",
                headers={"Authorization": f"Bearer {key}"},
                json={"prompt": "hello"},
            )
            assert resp.status_code == 200
        bad = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer unknown-key"},
            json={"prompt": "hello"},
        )
        assert bad.status_code == 401


def test_jwt_rs256_verification_claims_and_rejection_cases() -> None:
    priv_pem, pub_pem = _generate_rsa_pem_pair()
    other_priv_pem, _ = _generate_rsa_pem_pair()

    settings = load_settings(
        env={
            "NEXUS_AUTH_MODE": "jwt",
            "NEXUS_AUTH_JWT_PUBLIC_KEY_PEM": pub_pem,
            "NEXUS_AUTH_JWT_ISSUER": "https://idp.example.internal",
            "NEXUS_AUTH_JWT_AUDIENCE": "project-nexus",
        }
    )
    app = create_app(settings, configure_loggers=False)
    now = int(time.time())

    def make_token(claims: dict[str, Any], *, key: str = priv_pem) -> str:
        return jwt.encode(claims, key, algorithm="RS256")

    valid = make_token(
        {
            "sub": "svc-ml-client",
            "iss": "https://idp.example.internal",
            "aud": "project-nexus",
            "exp": now + 300,
        }
    )
    expired = make_token(
        {
            "sub": "svc-ml-client",
            "iss": "https://idp.example.internal",
            "aud": "project-nexus",
            "exp": now - 600,
        }
    )
    wrong_aud = make_token(
        {
            "sub": "svc-ml-client",
            "iss": "https://idp.example.internal",
            "aud": "wrong-service",
            "exp": now + 300,
        }
    )
    wrong_iss = make_token(
        {
            "sub": "svc-ml-client",
            "iss": "https://attacker.example",
            "aud": "project-nexus",
            "exp": now + 300,
        }
    )
    missing_sub = make_token(
        {
            "iss": "https://idp.example.internal",
            "aud": "project-nexus",
            "exp": now + 300,
        }
    )
    wrong_key = make_token(
        {
            "sub": "svc-ml-client",
            "iss": "https://idp.example.internal",
            "aud": "project-nexus",
            "exp": now + 300,
        },
        key=other_priv_pem,
    )
    # Forge an alg=none unsigned JWT
    header_b64 = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=")
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "admin",
                "iss": "https://idp.example.internal",
                "aud": "project-nexus",
                "exp": now + 300,
            }
        ).encode()
    ).rstrip(b"=")
    unsigned_none = f"{header_b64.decode()}.{payload_b64.decode()}."

    with TestClient(app) as client:
        ok = client.post(
            "/v1/generate",
            headers={"Authorization": f"Bearer {valid}"},
            json={"prompt": "jwt ok"},
        )
        assert ok.status_code == 200

        models_ok = client.get("/v1/models", headers={"Authorization": f"Bearer {valid}"})
        assert models_ok.status_code == 200
        assert client.get("/v1/models").status_code == 401

        for bad_tok in (expired, wrong_aud, wrong_iss, missing_sub, wrong_key, unsigned_none):
            rejected = client.post(
                "/v1/generate",
                headers={"Authorization": f"Bearer {bad_tok}"},
                json={"prompt": "should fail"},
            )
            assert rejected.status_code == 401


def test_jwt_es256_via_custom_jwks_key_resolver() -> None:
    priv_pem, pub_pem = _generate_ec_pem_pair()
    settings = load_settings(
        env={
            "NEXUS_AUTH_MODE": "jwt",
            "NEXUS_AUTH_JWT_JWKS_URL": "https://idp.example.internal/.well-known/jwks.json",
            "NEXUS_AUTH_JWT_AUDIENCE": "nexus-api",
        }
    )
    auth = GatewayAuthenticator(settings.auth, jwks_key_resolver=lambda _token: pub_pem)
    token = jwt.encode(
        {"sub": "team-a", "aud": "nexus-api", "exp": int(time.time()) + 120},
        priv_pem,
        algorithm="ES256",
    )
    caller = auth.authenticate(f"Bearer {token}")
    assert caller.auth_scheme == "jwt"
    assert caller.caller_id.startswith("jwt:")
    assert "team-a" not in caller.caller_id  # hashed for privacy


def test_auth_mode_config_validation() -> None:
    with pytest.raises(ConfigError, match="api_key"):
        load_settings(env={"NEXUS_AUTH_MODE": "api_key"})

    with pytest.raises(ConfigError, match=r"auth\.jwt"):
        load_settings(env={"NEXUS_AUTH_MODE": "jwt"})

    with pytest.raises(ConfigError, match="jwks_url"):
        load_settings(
            env={
                "NEXUS_AUTH_MODE": "jwt",
                "NEXUS_AUTH_JWT_JWKS_URL": "ftp://invalid",
            }
        )
