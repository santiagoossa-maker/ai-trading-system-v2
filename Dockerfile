# Multi-stage Docker build for AI Trading System V2
# Optimized for production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libta-lib-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRADING_ENV=production

# Create app user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libta-lib0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/backups /app/models
RUN chown -R trading:trading /app

# Switch to app user
USER trading

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["python", "-m", "src.api.trading_api"]

# Development stage
FROM production as development

ENV TRADING_ENV=development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

# Additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

USER trading

# Command for development
CMD ["python", "-m", "uvicorn", "src.api.trading_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]