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

# Copy application code. Explicit paths rather than `COPY . .`: the repo root
# also holds the 361 MB working index, the dashboard (a separate image) and the
# evaluation harness, none of which the API serves from.
COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Ship the measured index instead of baking a fresh one.
#
# Baking meant the deployment answered off a corpus no published number
# described — build-time ingest cannot use OpenAI embeddings without putting a
# key in the build, so the image pinned local MiniLM and produced its own
# 8,099-chunk corpus while every score in the README came from 8,232 OpenAI-
# embedded chunks. Two systems, one set of numbers.
#
# chroma_dist/ is that corpus, built and checked by scripts/build_index_dist.py
# (8,232 chunks, digest c2f8c13673cf5ca5). It lands at the default persist
# path, so nothing downstream has to know where the index came from.
COPY chroma_dist/ ./chroma_data/

# Prevent transformers from trying to load TensorFlow / Keras
ENV USE_TF=0
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# HuggingFace cache must be writable (appuser has no home dir by default)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Named explicitly even though it matches config.py's default. The shipped
# index is 1536-dimensional; MiniLM emits 384, so a deployment that overrode
# this would not fail loudly — it would query the wrong space. The image and
# its index have to agree, and this is where that agreement is written down.
#
# This is an embedding provider, not a generation one: LLM_PROVIDER is set at
# deploy time and the public demo runs Gemini's free tier. Embedding a question
# costs ~$0.000005 against text-embedding-3-small's $0.02/1M tokens, which is
# what makes an unauthenticated public URL affordable; generating an answer
# with gpt-4o costs ~$0.012, which is what makes it not.
ENV EMBEDDING_PROVIDER=openai

# The image serves a fixed corpus, so it does not accept uploads. Set in the
# image rather than at deploy time: the demo's admin password is published on
# the landing page, and a future deploy that forgot this flag would silently
# hand every visitor write access to the index the published numbers describe.
ENV ALLOW_RUNTIME_INGEST=false

# The cross-encoder is the one model still downloaded from HuggingFace, and it
# sits in the request path: without this it fetches on the first query rather
# than at build time, and the first visitor pays for it.
RUN python -c "from src.retrieval.reranker import get_reranker; get_reranker()"

# Create non-root user and hand over ownership of the baked data + caches
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/.cache/huggingface /app/data/edgar /app/chroma_data /app/audit_logs \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run sets $PORT (8080); EXPOSE is documentation only.
EXPOSE 8080

CMD ["python", "scripts/start.py"]
