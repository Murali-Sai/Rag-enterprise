# ── Build stage ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install CPU-only PyTorch first (saves ~1.5 GB vs CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project deps (no dev/eval extras)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# ── Runtime stage ───────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code + prebuilt data
COPY . .

# Prevent transformers from trying to load TensorFlow / Keras
ENV USE_TF=0
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["sh", "-c", "python scripts/seed_users.py && uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
