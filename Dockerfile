# syntax=docker/dockerfile:1
#
# smvis container for Google Cloud Run (or any Docker host).
# Cloud Run runs linux/amd64; the bundled nuXmv is a Linux x86_64 binary, so the
# platform is pinned to amd64 to keep the two in sync.

##############################################################################
# Stage 1 — extract the Linux nuXmv binary from the bundled tarball.
# Kept in a throwaway stage so neither the ~30 MB tarball nor xz-utils end up
# in the final image.
##############################################################################
FROM --platform=linux/amd64 debian:bookworm-slim AS nuxmv
RUN apt-get update \
    && apt-get install -y --no-install-recommends xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /extract
COPY bin/nuxmv/nuXmv-2.1.0-linux64.tar.xz ./
RUN tar -xJf nuXmv-2.1.0-linux64.tar.xz \
        nuXmv-2.1.0-linux64/bin/nuXmv --strip-components=2 \
    && chmod +x nuXmv

##############################################################################
# Stage 2 — runtime image.
##############################################################################
FROM --platform=linux/amd64 python:3.12-slim AS runtime

# nuXmv 2.1.0 ships as a statically linked binary (verified: `ldd` reports
# "not a dynamic executable"), so no extra system libraries are needed at runtime.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SMVIS_NUXMV_PATH=/app/bin/nuxmv/nuXmv \
    PORT=8080

WORKDIR /app

# Install the package as editable so the paths app.py derives from __file__
# (examples/ for models, bin/nuxmv/ for the binary) resolve exactly as they do
# in local development.
COPY pyproject.toml ./
COPY src/ ./src/
COPY examples/ ./examples/
RUN pip install --upgrade pip \
    && pip install -e ".[deploy]"

# Drop in the Linux nuXmv binary extracted in stage 1.
COPY --from=nuxmv /extract/nuXmv /app/bin/nuxmv/nuXmv

EXPOSE 8080

# A single worker keeps the in-process nuXmv interactive session and the
# SHA-256 result cache coherent (both are module-level globals); threads provide
# request concurrency for ~20-30 users. Cloud Run injects $PORT.
CMD exec gunicorn smvis.wsgi:server \
        --bind "0.0.0.0:${PORT:-8080}" \
        --workers 1 \
        --threads 8 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
