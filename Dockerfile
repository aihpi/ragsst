# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and use pyproject.toml for dependencies
COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv && \
    uv venv .venv && \
    uv sync

# Copy application code
COPY app.py ./
COPY ragsst ./ragsst

# Add metadata labels for GitHub Container Registry
LABEL org.opencontainers.image.source="https://github.com/aihpi/ragsst"
LABEL org.opencontainers.image.description="RAGSST - Retrieval Augmented Generation and Semantic-Search Tool"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.documentation="https://github.com/aihpi/ragsst/blob/main/README.md"

# Expose port
EXPOSE 7860

# Command to run the application
CMD [".venv/bin/python", "app.py"]