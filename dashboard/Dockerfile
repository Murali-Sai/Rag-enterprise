# Query dashboard — a thin Streamlit client for the API.
#
# Deliberately not built from the API image. That one carries torch,
# sentence-transformers and a baked vector index; the dashboard imports none
# of it and talks to the API over HTTP, so sharing the base would inherit a
# multi-GB image and a several-minute build for a process whose entire
# dependency set is streamlit plus requests.
#
# The build context is this directory, not the repo root, which is what lets
# `gcloud run deploy --source dashboard` work without a Cloud Build config.

FROM python:3.11-slim

WORKDIR /app

# Installed by name rather than from pyproject's `dashboard` extra, so this
# layer depends on nothing in the build context and is not invalidated by a
# source change. Keep the pins in step with that extra.
RUN pip install --no-cache-dir "streamlit>=1.40.0" "requests>=2.32.0"

COPY app.py ./dashboard/app.py

ENV PYTHONUNBUFFERED=1
# Where the API lives. Overridden by compose to the service name, and by the
# deploy environment to the API's public URL.
ENV RAG_API_URL=http://localhost:8000
# Cloud Run injects PORT and requires the container to listen on it; locally
# nothing sets it and 8501 is Streamlit's own default. Defaulted here rather
# than hardcoded in the CMD, which is what made the first deploy fail its
# health check.
ENV PORT=8501

RUN groupadd -r appuser && useradd -r -g appuser appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Shell form, so $PORT expands at run time. Streamlit binds loopback by
# default, which is unreachable from outside the container; headless
# suppresses the browser-open attempt and the email prompt that would
# otherwise block a non-interactive start.
CMD streamlit run dashboard/app.py \
      --server.address=0.0.0.0 \
      --server.port=${PORT} \
      --server.headless=true \
      --browser.gatherUsageStats=false
