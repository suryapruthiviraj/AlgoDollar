# syntax=docker/dockerfile:1.7
#
# AlgoDollar backend (FastAPI) — production image.
#
# Build context is ./backend (see .github/workflows/docker.yml and
# docker-compose.yml). Target platform is linux/amd64.
#
#   docker build -t algodollar/backend:local ./backend
#
# Design notes:
#   * Two stages. Dependencies are resolved into a self-contained virtualenv in
#     the builder; only /opt/venv crosses into the runtime image, so pip, its
#     caches and the C toolchain never ship.
#   * Runs as an unprivileged user (uid 10001). Nothing in the image is
#     writable by that user except /app/logs.
#   * No credentials are baked in. See .dockerignore — a stray .env would be
#     read by app/core/config.py and could silently flip TRADING_MODE.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

# Every pinned dependency currently resolves to a manylinux wheel, so this
# toolchain is normally unused. It is kept as insurance because the transitive
# dependencies (twisted, zope.interface, ... via kiteconnect) are unpinned and
# may drift to a version that has to be compiled. It stays in this stage.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than the system site-packages: it makes the runtime copy a
# single self-contained directory. Both stages share the same base image, so the
# interpreter symlinks inside /opt/venv stay valid.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the requirements file, so this layer caches until dependencies change.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Consumed from the CI build-args. A git SHA is not a secret; nothing sensitive
# may be passed as ARG/ENV, since both are readable via `docker history`.
ARG BUILD_SHA=unknown

LABEL org.opencontainers.image.title="AlgoDollar Backend" \
      org.opencontainers.image.description="AlgoDollar quantitative trading platform API" \
      org.opencontainers.image.source="https://github.com/algodollar/algodollar" \
      org.opencontainers.image.revision="${BUILD_SHA}"

# libgomp1 — required at run time by the LightGBM manylinux wheel (OpenMP).
#            Without it the build still succeeds but `import lightgbm` raises
#            "libgomp.so.1: cannot open shared object file" at startup.
# curl     — used by HEALTHCHECK below; slim images do not ship it.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

# Unprivileged service account. A fixed high uid/gid keeps bind-mount ownership
# predictable and cannot collide with a real host user. The home directory is
# created because some scientific libraries (joblib, matplotlib) fall back to
# writing caches under $HOME and would fail on a non-existent path.
RUN groupadd --system --gid 10001 appuser \
 && useradd --system --uid 10001 --gid appuser \
            --create-home --home-dir /home/appuser \
            --shell /usr/sbin/nologin appuser

# TRADING_MODE is a safety default, not configuration: the image never ships
# anything but paper trading. Going live must be a deliberate act at deploy
# time, never a property of the image. app/core/config.py independently defaults
# trading_mode to "paper", and docker-compose.yml sets it a third time.
# Nothing secret may ever be added here — ENV is readable via `docker history`.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    TRADING_MODE=paper

# Dependencies stay root-owned and read-only to the service account.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Only the application package. Tests, docs and tooling are excluded here and in
# .dockerignore; the test suite is never present in, nor run from, this image.
COPY --chown=appuser:appuser app/ ./app/

# docker-compose.yml mounts the `backend-logs` named volume at /app/logs. Docker
# seeds a fresh named volume from the image path, so this directory must exist
# and be owned by appuser or the volume mounts root-owned and unwritable.
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

USER appuser

EXPOSE 8000

# /api/v1/health is the real route: app.include_router(api_router,
# prefix="/api/v1") plus the health router at prefix "/health" with
# @router.get(""). Note it is NOT /health — that path 404s.
#
# Verified against the running app: the handler catches its own database and
# Redis errors, so it answers 200 with {"status":"degraded","db":"disconnected"}
# rather than failing when Postgres is not up yet. The probe therefore reports
# liveness, and the start period only has to cover interpreter startup plus the
# numpy/pandas/scikit-learn/lightgbm imports.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

# Single worker on purpose. The lifespan handler builds one execution stack and
# an in-process WebSocket connection registry on app.state; running multiple
# workers would create several independent order-management stacks in one
# deployment. Scale with replicas behind a load balancer, not with --workers.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-server-header"]
