# GenHealth: Multimodal Medical Report Analysis
# Production-ready Docker container

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_CACHE_DIR=/app/models \
    DATA_DIR=/app/data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/src /app/data /app/models /app/logs

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
RUN pip install -e .

# Copy source code
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY configs/ /app/configs/

# Create non-root user
RUN groupadd -r genhealth && useradd -r -g genhealth genhealth
RUN chown -R genhealth:genhealth /app
USER genhealth

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "genhealth.api.main"]