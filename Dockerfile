# =============================================================================
# InvestIQ — Dockerfile
# Multi-stage build: Python 3.13-slim, non-root user, health-check
# Phase 7: updated for Python 3.13, hardened security, HEALTHCHECK added
# =============================================================================

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies into a prefix we can copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Create non-root user (security hardening — Phase 7)
RUN groupadd -r investiq && useradd -r -g investiq -d /app -s /sbin/nologin investiq

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source code
COPY --chown=investiq:investiq . .

# Switch to non-root user
USER investiq

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Default: run FastAPI backend ──────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
