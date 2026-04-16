#!/bin/bash
set -e

echo "=========================================="
echo "RAG Enterprise - Demo Setup"
echo "=========================================="

echo ""
echo "[1/3] Seeding demo users..."
python scripts/seed_users.py

echo ""
echo "[2/3] Ingesting sample documents..."
python scripts/ingest_samples.py

echo ""
echo "[3/3] Demo environment ready!"
echo ""
echo "Start the server with: make dev"
echo ""
echo "Then try these commands:"
echo ""
echo "  # Login as HR user"
echo '  curl -s -X POST http://localhost:8000/auth/token \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"username":"hr_user","password":"hrpass1234!"}'"'"' | python -m json.tool'
echo ""
echo "  # Query with the token"
echo '  curl -s -X POST http://localhost:8000/query \'
echo '    -H "Authorization: Bearer <token>" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"question":"What is the PTO policy?"}'"'"' | python -m json.tool'
