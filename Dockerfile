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

# Expose port
EXPOSE 7860

# Command to run the application
CMD [".venv/bin/python", "app.py"]