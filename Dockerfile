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

# Install system deps for lxml (used by EDGAR parser)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Prevent transformers from trying to load TensorFlow / Keras
ENV USE_TF=0
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# HuggingFace cache must be writable (appuser has no home dir by default)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Bake the SEC filing index + embedding model into the image at build time.
# This downloads the 5 real 10-K filings, parses them, and ingests ~6,400
# chunks into ChromaDB during the build — so Cloud Run cold starts are instant
# instead of running a ~4-minute ingest on every container start.
RUN python scripts/download_filings.py \
    && python scripts/ingest_edgar.py --from-disk \
    && python -c "from src.ingestion.embeddings import get_embedding_model; get_embedding_model()"

# Create non-root user and hand over ownership of the baked data + caches
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/.cache/huggingface /app/data/edgar /app/chroma_data /app/audit_logs \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run sets $PORT (8080); EXPOSE is documentation only.
EXPOSE 8080

CMD ["python", "scripts/start.py"]
