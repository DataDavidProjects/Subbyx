# Subbyx - Fraud Detection System Context

Subbyx is a high-performance, real-time fraud detection system designed to analyze subscription checkout requests. It leverages a modern machine learning stack to provide predictive insights and automated decision-making.

## Project Overview

*   **Objective:** Predict if a customer subscription request (`mode=payment`) will result in "dunning" (>15 days late on installments).
*   **Backend:** FastAPI (Python 3.14) managing feature retrieval, model inference, and training pipelines.
*   **Frontend:** Next.js 16 (TypeScript) providing real-time and batch prediction dashboards.
*   **Machine Learning:** MLflow for model lifecycle management, Feast for feature store operations, and LightGBM/Scikit-learn for modeling.

## Architecture & Key Components

### Backend (`src/backend`)
*   **Routing:** Organized under `routes/`. The primary logic resides in `routes/fraud/`, which includes sub-routers for `checkout`, `score`, `decision`, `segment`, and `batch` operations.
*   **Inference Engine:** Located in `services/fraud/inference/`. Supports multi-model execution:
    *   **Production:** The primary model used for decisions.
    *   **Shadow:** An "always-on" model that scores requests for logging/comparison without affecting decisions.
    *   **Canary:** A traffic-gated model (e.g., 10% of traffic) that can take over decision-making for safe rollouts.
*   **Feature Store (Feast):** Managed in `feature_repo/`. Defines entities (`customer`) and feature views (`customers`, `charges`, `checkout_history`). Uses Redis as an online store in Docker and Parquet files for offline storage.
*   **Model Loading:** `ModelLoader` dynamically fetches models from MLflow using aliases (`@production`, `@shadow`, `@canary`).

### Frontend (`src/frontend`)
*   **Framework:** Next.js 16 with App Router.
*   **UI Components:** Built with Tailwind CSS, Radix UI, and Shadcn UI.
*   **Features:**
    *   **Real-time Prediction:** Interactive form to test checkout fraud scoring.
    *   **Batch Prediction:** Interface for uploading CSV files for bulk analysis.

### Data Pipeline (`scripts/`)
*   **ETL:** Scripts for cleaning raw CSVs, auditing data quality, and performing time-based train/test splits.
*   **Feature Engineering:** Aggregation scripts that compute historical customer behavior (e.g., charge failure rates).

## Development & Operations

### Key Commands (via `Makefile`)
*   **Setup:** `make install` (installs backend `uv` and frontend `npm` dependencies).
*   **Local Dev:** `make dev` (starts Backend on :8001, MLflow on :5002, and Frontend on :3001).
*   **Docker:** `make up` (starts all services including Redis and MLflow).
*   **Training:** `make train` (executes the training pipeline and registers the model in MLflow).
*   **Feature Store:** `make feast-apply` (registers features), `make feast-materialize` (populates online store), and `make feast-ui` (launches the dashboard).

### Configuration
*   **Inference Config:** `src/backend/services/fraud/inference/config.yaml` controls shadow/canary model URIs and traffic percentages.
*   **Environment Variables:** Managed via `.env` files in `frontend/` and environment-specific configs in `backend/`.

### Development Conventions
*   **Python Tooling:** Uses `uv` for lightning-fast dependency management and `ruff` for linting/formatting.
*   **Type Safety:** Strict Pydantic models for API schemas and TypeScript for the frontend.
*   **Compatibility:** Includes a custom SQLite converter in `main.py` to bridge Feast Unix epochs and Python 3.14/IPython ISO timestamps.

## Dataset Entities
*   **Customers:** Core entity with anonymized fiscal/PII similarity scores.
*   **Checkouts:** Subscription requests. `mode=payment` is the primary target for prediction.
*   **Charges/Payment Intents:** Transaction history used to derive behavioral features (e.g., `n_failures`).
