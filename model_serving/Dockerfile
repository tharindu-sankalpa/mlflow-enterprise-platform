# Multi-stage build for smaller final image
FROM python:3.10-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to use system Python
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main --no-root

# Production stage
FROM python:3.10-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlserver
USER mlserver
WORKDIR /home/mlserver

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories
RUN mkdir -p /home/mlserver/models

# Set environment variables for MLServer
ENV MLSERVER_MODEL_IMPLEMENTATION=mlserver_mlflow.MLflowRuntime
ENV MLSERVER_HTTP_PORT=8080
ENV MLSERVER_GRPC_PORT=9000
ENV MLSERVER_MODELS_DIR=/mnt/models
ENV PYTHONPATH=/home/mlserver
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/v2/health/ready || exit 1

# Switch back to root for final setup
USER root

# Expose ports
EXPOSE 8080 8082 9000

# Switch to mlserver user for runtime
USER mlserver

# Start MLServer
CMD ["mlserver", "start", "/mnt/models"]