# Multi-stage Dockerfile for BERT Sentiment Analysis Training Pipeline
# This Dockerfile automates: data download, preprocessing, and model training

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application setup and training
FROM base as training

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Create necessary directories
RUN mkdir -p data/raw \
    data/processed \
    src/data \
    src/models \
    src/utils \
    models/checkpoints \
    mlruns \
    logs \
    cache

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create __init__.py files
RUN touch src/__init__.py \
    src/data/__init__.py \
    src/models/__init__.py \
    src/utils/__init__.py

# Create entrypoint script for automated pipeline
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "BERT Sentiment Analysis Training Pipeline"\n\
echo "========================================"\n\
echo ""\n\
\n\
# Step 1: Data Preparation\n\
echo "📥 Step 1/3: Downloading and preprocessing data..."\n\
python src/data/prepare_data.py\n\
echo "✅ Data preparation complete!"\n\
echo ""\n\
\n\
# Step 2: Model Training\n\
echo "🚀 Step 2/3: Starting model training..."\n\
python src/models/train.py\n\
echo "✅ Model training complete!"\n\
echo ""\n\
\n\
# Step 3: Summary\n\
echo "📊 Step 3/3: Training Summary"\n\
echo "========================================"\n\
if [ -f "models/checkpoints/best_model.pth" ]; then\n\
    echo "✅ Model saved: models/checkpoints/best_model.pth"\n\
    MODEL_SIZE=$(du -h models/checkpoints/best_model.pth | cut -f1)\n\
    echo "📦 Model size: $MODEL_SIZE"\n\
fi\n\
\n\
if [ -d "mlruns" ]; then\n\
    RUN_COUNT=$(find mlruns -type d -name "run-*" 2>/dev/null | wc -l)\n\
    echo "📈 MLflow runs: $RUN_COUNT"\n\
fi\n\
\n\
echo ""\n\
echo "========================================"\n\
echo "🎉 Pipeline completed successfully!"\n\
echo "========================================"\n\
echo ""\n\
echo "Next steps:"\n\
echo "1. View MLflow UI: mlflow ui --host 0.0.0.0"\n\
echo "2. Access trained model: models/checkpoints/best_model.pth"\n\
echo "3. Check logs: logs/"\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Set the entrypoint
ENTRYPOINT ["/app/run_pipeline.sh"]

# Default command (can be overridden)
CMD []

# Expose MLflow UI port
EXPOSE 5000

# Volume mounts for persistence
VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/logs"]