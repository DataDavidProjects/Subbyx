# =============================================================================
# Subbyx Makefile
# =============================================================================

.PHONY: help install sync build up down restart logs clean-containers lint format \
        clean-data data-audit data-elaborate data-features data-process \
        redis feast-apply feast-materialize feast-restart feast-status feast-ui feast-verify \
        dev-backend dev-frontend dev mlflow model-register

# Default dates for materialize
START ?= 2024-01-01
END ?= 2025-01-01

# Default Redis host for Feast online store (overridden by docker-compose)
export FEAST_REDIS_HOST ?= localhost:6379

# -----------------------------------------------------------------------------
# Setup & Installation
# -----------------------------------------------------------------------------

help:
	@echo "Subbyx - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install all dependencies (backend + scripts + frontend)"
	@echo "  make sync             - Re-sync Python deps (backend + scripts)"
	@echo "  make build            - Build Docker images"
	@echo ""
	@echo "Docker (all services):"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - Show live logs"
	@echo "  make clean-containers - Remove containers and volumes"
	@echo "  make redis           - Start local Redis on port 6379"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make clean-data       - Clean raw CSV data"
	@echo "  make data-audit      - Run data quality audit"
	@echo "  make data-elaborate  - Split data into train/test"
	@echo "  make data-process    - Convert CSV to Parquet"
	@echo "  make data-aggregates - Compute aggregated features"
	@echo ""
	@echo "Feature Store:"
	@echo "  make feast-apply             - Apply feature definitions to registry"
	@echo "  make feast-materialize      - Materialize features to Redis online store"
	@echo "  make feast-materialize START=2024-01-01 END=2025-01-01 - Custom dates"
	@echo "  make feast-restart          - Apply + materialize"
	@echo "  make feast-restart START=2024-01-01 END=2025-01-01 - Custom dates"
	@echo "  make feast-status           - Show registered features"
	@echo "  make feast-verify           - Check Redis has feature keys"
	@echo "  make feast-ui               - Start Feast UI on http://localhost:8888"
	@echo ""
	@echo "Feature Selection:"
	@echo "  make feature-selection      - Run feature selection pipeline"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linters"
	@echo "  make format          - Format code with ruff"
	@echo ""
	@echo "Local Development (no Docker):"
	@echo "  make dev-backend     - Backend on port 8001 + MLflow on 5002"
	@echo "  make dev-frontend    - Frontend on port 3001"
	@echo "  make dev             - Both backend and frontend locally"
	@echo "  make mlflow          - MLflow only on port 5002"
	@echo "  make model-register  - Register dummy model in MLflow"

# Install all project dependencies
install:
	cd src/backend && uv sync
	cd scripts && uv sync
	cd src/frontend && npm install

# Re-sync Python dependencies (backend + scripts)
sync:
	cd src/backend && uv sync
	cd scripts && uv sync

# Build Docker images
build: install
	cd docker && docker compose build --build-arg PYTHON_VERSION=$(shell cat src/backend/.python-version)

# -----------------------------------------------------------------------------
# Docker Services
# -----------------------------------------------------------------------------

# Start all services (frontend, backend, redis, mlflow)
up: build
	cd docker && docker compose up -d
	@echo ""
	@echo "Services are running:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  MLflow:   http://localhost:5002"

# Stop all services
down:
	cd docker && docker compose down

# Restart all services
restart:
	cd docker && docker compose restart

# Show live logs
logs:
	cd docker && docker compose logs -f

# Remove containers and volumes
clean-containers:
	cd docker && docker compose down -v
	docker system prune -f

# Start local Redis (tries docker, falls back to redis-server)
redis:
	@echo "Stopping any existing process on port 6379..."
	@lsof -ti:6379 2>/dev/null | xargs kill -9 2>/dev/null || true
	@docker rm -f subbyx-redis-local 2>/dev/null || true
	@if docker info >/dev/null 2>&1; then \
		docker run -d --name subbyx-redis-local -p 6379:6379 redis:7-alpine; \
	elif command -v redis-server >/dev/null 2>&1; then \
		redis-server --daemonize yes --port 6379; \
	else \
		echo "Error: neither Docker nor redis-server found. Install one of them."; \
		exit 1; \
	fi
	@echo "Redis running on localhost:6379"

# -----------------------------------------------------------------------------
# Data Pipeline
# -----------------------------------------------------------------------------

# Clean raw CSV files -> data/01-clean/
clean-data:
	cd scripts && uv run python data/clean.py

# Run data quality audit notebooks -> docs/data/
data-audit: clean-data
	cd scripts && VIRTUAL_ENV= uv run python notebooks/01-customers-audit.py
	cd scripts && VIRTUAL_ENV= uv run python notebooks/02-checkouts-audit.py
	cd scripts && VIRTUAL_ENV= uv run python notebooks/03-charges-audit.py
	cd scripts && VIRTUAL_ENV= uv run python notebooks/04-payment-intents-audit.py
	cd scripts && VIRTUAL_ENV= uv run python notebooks/05-addresses-audit.py
	cd scripts && VIRTUAL_ENV= uv run python notebooks/06-stores-audit.py

# Time-based train/test split -> data/02-elaboration/
data-elaborate:
	cd scripts && uv run python data/elaborate.py

# Convert CSV to Parquet + type fixes -> data/03-processed/
data-process:
	cd scripts && .venv/bin/python 01_csv_to_parquet.py

# Compute aggregated features -> data/04-modeling/
data-aggregates:
	cd scripts && uv run python compute_aggregates.py

# -----------------------------------------------------------------------------
# Feature Store (Feast)
# -----------------------------------------------------------------------------

# Register feature definitions in Feast registry
feast-apply:
	cd src/backend/feature_repo && uv run feast apply

# Materialize features to Redis online store
# Usage: make feast-materialize START=2024-01-01 END=2025-01-01
feast-materialize:
	cd src/backend/feature_repo && uv run feast materialize $(START) $(END)

# Restart feature store (apply + materialize)
# Usage: make feast-restart START=2024-01-01 END=2025-01-01
feast-restart:
	cd src/backend/feature_repo && uv run feast apply && uv run feast materialize $(START) $(END)

# Show registered feature views
feast-status:
	cd src/backend/feature_repo && uv run feast registry-summary

# Verify features are stored in Redis
feast-verify:
	@echo "Checking Redis for Feast feature keys..."
	@docker exec subbyx-redis-local redis-cli keys '*' 2>/dev/null | head -20 || redis-cli keys '*' | head -20
	@echo ""
	@echo "Total keys: $$(docker exec subbyx-redis-local redis-cli dbsize 2>/dev/null || redis-cli dbsize)"

# Run feature selection pipeline
feature-selection:
	cd scripts && uv run python feature_selection.py

# Start Feast UI
feast-ui:
	@echo "Stopping any existing process on port 8888..."
	@lsof -ti:8888 2>/dev/null | xargs kill -9 2>/dev/null || true
	cd src/backend/feature_repo && uv run feast ui

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

# Run linters (ruff + eslint)
lint:
	cd src/backend && uv run ruff check .
	cd src/frontend && npm run lint

# Format code with ruff
format:
	cd src/backend && uv run ruff format .

# -----------------------------------------------------------------------------
# Local Development (no Docker)
# -----------------------------------------------------------------------------

# Backend + MLflow (requires frontend/.env.local with NEXT_PUBLIC_API_URL=http://localhost:8001)
dev-backend: redis
	@echo "Stopping any existing processes on ports 8001, 5002..."
	@lsof -ti:8001 -ti:5002 2>/dev/null | xargs kill -9 2>/dev/null || true
	sleep 2
	@echo "Starting MLflow on http://localhost:5002..."
	cd src/backend && uv run mlflow server --port 5002 \
		--backend-store-uri sqlite:////Users/davidelupis/Desktop/Subbyx/data/mlflow/mlflow.db \
		--default-artifact-root /Users/davidelupis/Desktop/Subbyx/data/mlflow/artifacts & \
	sleep 3
	@echo "Starting backend on http://localhost:8001..."
	cd src/backend && uv run python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001

# Frontend only
dev-frontend:
	@echo "Stopping any existing process on port 3001..."
	@lsof -ti:3001 2>/dev/null | xargs kill -9 2>/dev/null || true
	@echo "Clearing Next.js cache..."
	rm -rf src/frontend/.next
	@echo "Starting frontend on http://localhost:3001..."
	@echo "Press Ctrl+C to stop"
	cd src/frontend && npm run dev -- -p 3001

# Both backend and frontend locally
dev: dev-backend dev-frontend

# MLflow tracking server only
mlflow:
	@echo "Starting MLflow on http://localhost:5002..."
	@echo "Press Ctrl+C to stop"
	cd src/backend && uv run mlflow server --port 5002 \
		--backend-store-uri sqlite:////Users/davidelupis/Desktop/Subbyx/data/mlflow/mlflow.db \
		--default-artifact-root /Users/davidelupis/Desktop/Subbyx/data/mlflow/artifacts

# Register dummy model in MLflow for testing
model-register:
	@echo "Registering dummy fraud-detector model in MLflow..."
	cd scripts && VIRTUAL_ENV= uv run python register_dummy_model.py
	@echo "Done. Model registered as fraud-detector@production"

# Train model on real data (creates training data + trains)
train: mlflow
	@echo "Creating training data from data/01-clean..."
	cd scripts && uv run python create_training_data.py
	@echo "Training fraud detection model..."
	cd scripts && uv run python train_simple.py
	@echo "Done. Model trained and registered as fraud-detector@production"

# Create training data only (no training)
create-training-data:
	cd scripts && uv run python create_training_data.py
