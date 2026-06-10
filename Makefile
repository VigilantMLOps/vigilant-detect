.PHONY: install up down logs test test-fast lint generate train train-script deploy rollback seed run run-prod help

install:
	poetry install

# ── Infrastructure ─────────────────────────────────────────────────────────────

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

# ── Development ────────────────────────────────────────────────────────────────

test:
	poetry run pytest tests/ -v --tb=short -x

test-fast:
	poetry run pytest tests/ -v --tb=short -x -k "not adversarial"

lint:
	poetry run python -m py_compile main.py core/**/*.py services/*.py api/v1/*.py

# ── Data & Training ────────────────────────────────────────────────────────────

generate:
	poetry run python scripts/seed.py

train:
	docker-compose exec vigilant-detect poetry run python -m cli.main train

train-script:
	poetry run python scripts/train_ieee.py

# ── Model Lifecycle ────────────────────────────────────────────────────────────

deploy:
	@test -n "$(MODEL_ID)" || (echo "Usage: make deploy MODEL_ID=<model_id>" && exit 1)
	poetry run python -m cli.main deploy $(MODEL_ID)

rollback:
	@test -n "$(MODEL_ID)" || (echo "Usage: make rollback MODEL_ID=<model_id>" && exit 1)
	poetry run python -m cli.main rollback $(MODEL_ID)

seed:
	docker-compose exec vigilant-detect poetry run python scripts/seed.py

# ── Service ────────────────────────────────────────────────────────────────────

run:
	poetry run uvicorn main:app --host 0.0.0.0 --port 8001 --reload

run-prod:
	poetry run uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up              Start Redis / PostgreSQL / ClickHouse (docker-compose)"
	@echo "  make down            Stop infrastructure"
	@echo "  make logs            Tail docker-compose logs"
	@echo ""
	@echo "Development:"
	@echo "  make install         Install dependencies (poetry)"
	@echo "  make test            Run full test suite"
	@echo "  make test-fast       Run tests excluding adversarial suite"
	@echo "  make lint            Syntax check"
	@echo ""
	@echo "Data & Training:"
	@echo "  make seed            Full bootstrap: generate → train → deploy (needs DB)"
	@echo "  make train           Train via CLI (needs DB)"
	@echo "  make train-script    Train to disk only, no DB required"
	@echo ""
	@echo "Model Lifecycle:"
	@echo "  make deploy MODEL_ID=<id>    Promote model to production"
	@echo "  make rollback MODEL_ID=<id>  Roll back to a previous model"
	@echo ""
	@echo "Service:"
	@echo "  make run             Start service on :8001 (dev, auto-reload)"
	@echo "  make run-prod        Start service on :8001 (production)"
	@echo ""
	@echo "Typical first-time flow:"
	@echo "  make install && make up && make seed && make run"
	@echo ""
