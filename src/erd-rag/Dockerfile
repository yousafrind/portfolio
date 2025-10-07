# Use a small Python base
FROM python:3.11-slim

# Metadata
LABEL maintainer="you <you@example.com>" \
      description="ERD-RAG - Embedding + RAG for ERD schemas"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system deps required for faiss, building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home appuser
WORKDIR /app
USER appuser

# Copy files (use a thin copy to leverage Docker cache)
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY --chown=appuser:appuser . /app

# Expose a workdir and default command
ENV CHROMA_DIR=/app/.chroma
CMD ["python", "-m", "src.cli"]

