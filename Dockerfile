# Stage 1: Builder
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libnccl-dev \
    libnccl2 \
    libnvidia-ml-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . /build

RUN python3 setup.py build_ext --inplace

# Stage 2: Runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libnccl2 \
    libnvidia-ml-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/*.so /app/
COPY --from=builder /build/core/*.py /app/
COPY --from=builder /build/nexus_orchestrator.py /app/

EXPOSE 8000

CMD ["python3", "nexus_orchestrator.py"]