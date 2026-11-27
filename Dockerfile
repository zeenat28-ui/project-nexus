# Multi-stage CPU-only container image for the Project Nexus serving gateway.
# Production deployments should additionally pin base images by SHA256 digest.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --prefix=/install .

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="project-nexus" \
      org.opencontainers.image.description="LLM serving integration and operations gateway" \
      org.opencontainers.image.version="0.4.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/zeenat28-ui/project-nexus"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXUS_SERVER_HOST=0.0.0.0 \
    NEXUS_SERVER_PORT=8000 \
    NEXUS_METRICS_MODE=separate \
    NEXUS_METRICS_HOST=127.0.0.1 \
    NEXUS_METRICS_PORT=9090

RUN groupadd --gid 10001 nexus \
    && useradd --uid 10001 --gid 10001 --shell /usr/sbin/nologin --create-home nexus

COPY --from=builder /install /usr/local
COPY config/nexus.mock.yaml /etc/nexus/nexus.mock.yaml

WORKDIR /app
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

ENTRYPOINT ["nexus"]
CMD ["serve"]
