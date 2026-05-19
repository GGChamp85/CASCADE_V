FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4.0,<2.6" "torchaudio>=2.4.0,<2.6"

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY server/ ./server/
COPY scripts/ ./scripts/

RUN pip install ".[server]"

COPY data/ ./data/
COPY models/ ./models/
COPY outputs/ ./outputs/

EXPOSE 8766

CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8766"]
