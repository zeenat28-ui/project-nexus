"""Authentication and caller identification for Project Nexus.

Supports:
- ``none``: unauthenticated (for local development or when enforced externally by an edge proxy)
- ``api_key``: constant-time validation against one or more rotatable bearer tokens
- ``jwt``: asymmetric RS256/ES256 JWT validation via JWKS URL or static PEM public key
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import InvalidTokenError, PyJWKClient

from project_nexus.config import AuthSettings


class AuthenticationError(Exception):
    """Raised when an incoming request fails authentication."""

    def __init__(self, message: str = "Missing or invalid bearer token.") -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class AuthenticatedCaller:
    """Safe, non-secret identity of the caller for rate limiting and audit metadata."""

    caller_id: str
    auth_scheme: str


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Parse ``Authorization: Bearer <token>`` safely without raising."""
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def constant_time_bearer_matches(authorization_header: str | None, expected_secret: str) -> bool:
    """Validate a single bearer token in constant time."""
    token = extract_bearer_token(authorization_header)
    if token is None:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected_secret.encode("utf-8"))


def _hash_identifier(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class GatewayAuthenticator:
    """Validates caller credentials according to configured ``AuthSettings``."""

    def __init__(
        self,
        settings: AuthSettings,
        *,
        jwks_key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._mode = settings.resolved_mode()
        self._api_keys: tuple[str, ...] = tuple(
            secret.get_secret_value() for secret in settings.all_api_keys()
        )
        self._jwks_client: PyJWKClient | None = None
        self._jwks_key_resolver = jwks_key_resolver
        self._static_public_key: str | None = None

        if self._mode == "jwt" and settings.jwt is not None:
            if settings.jwt.public_key_pem is not None:
                self._static_public_key = settings.jwt.public_key_pem
            elif settings.jwt.jwks_url is not None and jwks_key_resolver is None:
                self._jwks_client = PyJWKClient(
                    settings.jwt.jwks_url,
                    cache_keys=True,
                    lifespan=settings.jwt.jwks_cache_seconds,
                    timeout=settings.jwt.jwks_timeout_seconds,
                )

    @property
    def mode(self) -> str:
        """Resolved active authentication mode: ``none``, ``api_key``, or ``jwt``."""
        return self._mode

    @property
    def is_enforced(self) -> bool:
        return self._mode != "none"

    def authenticate(
        self,
        authorization_header: str | None,
        *,
        client_host: str | None = None,
    ) -> AuthenticatedCaller:
        """Authenticate an incoming request or raise ``AuthenticationError``."""
        if self._mode == "none":
            host_id = client_host or "unknown"
            return AuthenticatedCaller(
                caller_id=f"ip:{_hash_identifier(host_id)}",
                auth_scheme="none",
            )

        token = extract_bearer_token(authorization_header)
        if token is None:
            raise AuthenticationError("Missing or invalid bearer token.")

        if self._mode == "api_key":
            return self._authenticate_api_key(token)

        if self._mode == "jwt":
            return self._authenticate_jwt(token)

        raise AuthenticationError("Unsupported authentication mode.")

    def _authenticate_api_key(self, token: str) -> AuthenticatedCaller:
        token_bytes = token.encode("utf-8")
        matched_index: int | None = None
        # Compare against every configured key without short-circuiting to reduce timing leaks.
        for idx, candidate in enumerate(self._api_keys):
            if (
                hmac.compare_digest(token_bytes, candidate.encode("utf-8"))
                and matched_index is None
            ):
                matched_index = idx
        if matched_index is None:
            raise AuthenticationError("Missing or invalid bearer token.")
        return AuthenticatedCaller(
            caller_id=f"key:{_hash_identifier(self._api_keys[matched_index])}",
            auth_scheme="api_key",
        )

    def _authenticate_jwt(self, token: str) -> AuthenticatedCaller:
        jwt_cfg = self._settings.jwt
        if jwt_cfg is None:  # pragma: no cover - prevented by config validation
            raise AuthenticationError("JWT authentication is not configured.")

        try:
            signing_key: Any
            if self._static_public_key is not None:
                signing_key = self._static_public_key
            elif self._jwks_key_resolver is not None:
                signing_key = self._jwks_key_resolver(token)
            elif self._jwks_client is not None:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
            else:  # pragma: no cover - prevented by config validation
                raise AuthenticationError("No JWT verification key source configured.")

            claims = jwt.decode(
                token,
                signing_key,
                algorithms=list(jwt_cfg.algorithms),
                audience=jwt_cfg.audience,
                issuer=jwt_cfg.issuer,
                leeway=jwt_cfg.leeway_seconds,
                options={
                    "require": ["exp", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_aud": jwt_cfg.audience is not None,
                    "verify_iss": jwt_cfg.issuer is not None,
                },
            )
        except InvalidTokenError as exc:
            raise AuthenticationError("Invalid or expired JWT bearer token.") from exc
        except Exception as exc:
            raise AuthenticationError("Unable to verify JWT bearer token.") from exc

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise AuthenticationError("JWT token missing non-empty 'sub' claim.")

        return AuthenticatedCaller(
            caller_id=f"jwt:{_hash_identifier(subject.strip())}",
            auth_scheme="jwt",
        )
