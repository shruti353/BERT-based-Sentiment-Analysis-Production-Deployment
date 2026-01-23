# Complete BERT Sentiment Analysis Pipeline for Windows
# Single Dockerfile that handles everything automatically

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p data/raw \
    data/processed \
    src/data \
    src/models \
    src/utils \
    models/checkpoints \
    mlruns \
    logs \
    cache

# Copy all source code
COPY src/ ./src/
COPY configs/ ./configs/ 2>/dev/null || :

# Create __init__.py files
RUN touch src/__init__.py \
    src/data/__init__.py \
    src/models/__init__.py \
    src/utils/__init__.py

# Create the pipeline execution script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "BERT Sentiment Analysis Training Pipeline"\n\
echo "========================================"\n\
echo "Running on Windows Docker"\n\
echo ""\n\
\n\
# Check if data already exists\n\
if [ -f "data/processed/train.csv" ]; then\n\
    echo "✅ Processed data found, skipping download..."\n\
else\n\
    echo "📥 Step 1/3: Downloading and preprocessing data..."\n\
    echo "This will take approximately 5 minutes..."\n\
    python src/data/prepare_data.py\n\
    echo "✅ Data preparation complete!"\n\
fi\n\
\n\
echo ""\n\
echo "🚀 Step 2/3: Starting model training..."\n\
echo "This will take approximately 30-40 minutes on CPU..."\n\
python src/models/train.py\n\
echo "✅ Model training complete!"\n\
\n\
echo ""\n\
echo "📊 Step 3/3: Training Summary"\n\
echo "========================================"\n\
\n\
if [ -f "models/checkpoints/best_model.pth" ]; then\n\
    MODEL_SIZE=$(du -h models/checkpoints/best_model.pth | cut -f1)\n\
    echo "✅ Model saved successfully!"\n\
    echo "📍 Location: models/checkpoints/best_model.pth"\n\
    echo "📦 Size: $MODEL_SIZE"\n\
fi\n\
\n\
echo ""\n\
echo "📈 MLflow Tracking:"\n\
if [ -d "mlruns/0" ]; then\n\
    echo "✅ Experiments tracked in mlruns/"\n\
    echo "💡 To view: docker run -p 5000:5000 -v %cd%/mlruns:/app/mlruns bert-sentiment mlflow ui --host 0.0.0.0"\n\
fi\n\
\n\
echo ""\n\
echo "========================================"\n\
echo "🎉 Pipeline completed successfully!"\n\
echo "========================================"\n\
echo ""\n\
echo "Your trained model is ready in:"\n\
echo "  → models/checkpoints/best_model.pth"\n\
echo ""\n\
echo "Next steps:"\n\
echo "1. View results in MLflow UI"\n\
echo "2. Use the model for inference"\n\
echo "3. Build FastAPI deployment"\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Set the entrypoint
ENTRYPOINT ["/app/run_pipeline.sh"]

# Expose MLflow port
EXPOSE 5000

# Define volumes for Windows compatibility
VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/logs"]