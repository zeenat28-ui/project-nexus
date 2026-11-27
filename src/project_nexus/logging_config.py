"""Structured logging utilities for Project Nexus.

Prompts, completions, raw request bodies, and credentials must never be emitted
into log records.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_SAFE_EXTRA_FIELDS: tuple[str, ...] = (
    "request_id",
    "method",
    "path",
    "status_code",
    "latency_ms",
    "backend",
    "outcome",
    "error_code",
    "error_type",
    "circuit_state",
    "host",
    "port",
    "metrics_mode",
)


def sanitize_request_id(candidate: str | None) -> str:
    """Return a safe request ID, generating a new UUID4 when missing or invalid."""
    if candidate:
        trimmed = candidate.strip()
        if _REQUEST_ID_PATTERN.fullmatch(trimmed):
            return trimmed
    return str(uuid.uuid4())


class JsonLogFormatter(logging.Formatter):
    """Single-line JSON log formatter with an explicit allowlist of structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in _SAFE_EXTRA_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__

        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class TextLogFormatter(logging.Formatter):
    """Human-readable structured key=value log formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [f"{ts} {record.levelname} {record.name}: {record.getMessage()}"]
        for field_name in _SAFE_EXTRA_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                parts.append(f"{field_name}={value}")
        return " ".join(parts)


def configure_logging(
    *,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_format: Literal["json", "text"] = "json",
) -> logging.Logger:
    """Configure the ``project_nexus`` logger namespace."""
    logger = logging.getLogger("project_nexus")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter: logging.Formatter = (
        JsonLogFormatter() if log_format == "json" else TextLogFormatter()
    )
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
