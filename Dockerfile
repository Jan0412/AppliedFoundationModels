# syntax=docker/dockerfile:1

# uv on a slim Debian base. No CUDA base image: the PyPI torch cu12 wheels bundle
# the CUDA runtime + cuDNN, so the host only needs the driver + the NVIDIA
# container toolkit. That keeps the image several GB smaller.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git         — vggt is a git dependency in uv.lock.
# libgl1,     — opencv-python (transitive via vggt) dlopen's libGL / libglib at
# libglib2.0-0  import time even for headless use.
# (No ffmpeg: the PyAV wheels bundle their own FFmpeg libraries.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/hf-cache

# Resolve dependencies first, from the lockfile only, so this layer is cached
# until the lock changes rather than on every source edit.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY config.yaml ./
COPY src/ src/

EXPOSE 8080

# viser binds 0.0.0.0 by default, so the mapped port is reachable from the host.
CMD ["uv", "run", "--no-dev", "python", "-m", "src.ui.app"]
