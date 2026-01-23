FROM python:3.12-slim AS base

RUN pip install uv && apt-get update && apt-get install -y git

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
#COPY .dvc/config .dvc/config

RUN uv sync --frozen --no-install-project

COPY src src/

# Copy model checkpoint(s) directly into the image (we'll include the trained model in `models/`)
COPY models models/

RUN uv sync --frozen

# Initialize git so DVC works (needed for dvc pull)
RUN git init && git config user.email "docker@build" && git config user.name "Docker Build"

# Copy DVC tracking files
#COPY data/raw/*.dvc data/raw/
#COPY data/processed/*.dvc data/processed/
#COPY models/*.dvc models/

# Skipping `dvc pull` in the Docker build; model checkpoints are copied directly into `models/`.
#RUN uv run dvc pull

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uv run uvicorn src.project.backend:app --host 0.0.0.0 --port ${PORT:-8080}"]
