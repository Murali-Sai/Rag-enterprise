#!/bin/sh
set -e

echo "=== Seeding users ==="
python scripts/seed_users.py || echo "WARNING: seed_users.py failed (users may already exist — continuing)"

echo "=== Starting uvicorn on port ${PORT:-8000} ==="
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
