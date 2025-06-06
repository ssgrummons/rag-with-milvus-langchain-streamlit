# frontend/Dockerfile
FROM python:3.12-slim

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

COPY src/ .

RUN poetry install --without dev

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -m -g appuser -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]