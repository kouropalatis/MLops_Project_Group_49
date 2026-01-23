FROM python:3.12-slim AS base

RUN pip install uv && apt-get update && apt-get install -y git

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY .dvc .dvc

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

# Initialize git so DVC works (needed for dvc pull)
RUN git init && git config user.email "docker@build" && git config user.name "Docker Build"

# Remove local config that causes issues in container
RUN rm -f .dvc/config.local

RUN uv run dvc pull

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uv run uvicorn src.project.backend:app --host 0.0.0.0 --port ${PORT:-8080}"]
