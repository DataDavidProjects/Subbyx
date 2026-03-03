ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY src/backend/pyproject.toml .
COPY src/backend/.python-version .
COPY src/backend/uv.lock .
RUN uv sync --frozen --no-dev

COPY src/backend/ .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
