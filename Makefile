.PHONY: install dev test lint format eval docker-up docker-down seed ingest demo download-filings ingest-edgar

install:
	pip install -e ".[dev,eval]"

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

eval:
	python -m evaluation.run_evaluation

seed:
	python scripts/seed_users.py

ingest:
	python scripts/ingest_samples.py

# SEC EDGAR commands
download-filings:
	python scripts/download_filings.py

ingest-edgar:
	python scripts/ingest_edgar.py --from-disk

# Full demo: seed users, download real SEC filings, ingest everything
demo: seed download-filings ingest-edgar ingest
	@echo ""
	@echo "Demo environment ready with real SEC EDGAR filings!"
	@echo "Run 'make dev' to start the server."
	@echo ""
	@echo "Try: curl -X POST http://localhost:8000/auth/token -H 'Content-Type: application/json' -d '{\"username\":\"research_analyst\",\"password\":\"research1!\"}'"

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down -v
