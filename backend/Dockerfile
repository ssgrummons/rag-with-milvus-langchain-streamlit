# === Stage 1: Build + download model ===
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install poetry==1.8.3


ARG EMBEDDING_MODEL
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    HF_HUB_CACHE="/.cache" \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN poetry install --without dev --no-root

RUN pip install huggingface_hub hf_xet

# Pre-download model while online
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/${EMBEDDING_MODEL}')" 

# === Stage 2: Offline run image ===
FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_CACHE="/.cache" 

COPY --from=builder /.cache /.cache

WORKDIR /app
COPY --from=builder /app /app
COPY src/ .

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -m -g appuser -u 1000 appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /.cache

USER appuser

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
