# ================================
# BERT Sentiment Analysis – Docker
# Windows-safe, CPU training
# ================================

FROM python:3.10-slim

WORKDIR /app

# ----------------
# System setup
# ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------
# Environment vars
# ----------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# ----------------
# Python deps
# ----------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------
# Project structure
# ----------------
RUN mkdir -p \
    data/raw \
    data/processed \
    models/checkpoints \
    mlruns \
    logs \
    cache \
    src/data \
    src/models \
    src/utils

# ----------------
# Copy source code
# ----------------
COPY src/ ./src/

# ----------------
# Init files
# ----------------
RUN touch \
    src/__init__.py \
    src/data/__init__.py \
    src/models/__init__.py \
    src/utils/__init__.py

# ----------------
# Pipeline runner
# ----------------
RUN cat <<'EOF' > /app/run_pipeline.sh
#!/bin/bash
set -e

echo "========================================"
echo "BERT Sentiment Analysis – Docker Pipeline"
echo "========================================"

python - <<PYCODE
import torch
print("CUDA available:", torch.cuda.is_available())
PYCODE

if [ ! -f data/processed/train.csv ]; then
  echo "📥 Preparing data..."
  python src/data/prepare_data.py
else
  echo "✅ Data already prepared"
fi

echo "🚀 Training model..."
python src/models/train.py

echo "========================================"
echo "🎉 Pipeline completed successfully"
echo "Model location: models/checkpoints"
echo "MLflow runs: mlruns/"
echo "========================================"
EOF

RUN sed -i 's/\r$//' /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

CMD ["/bin/bash", "/app/run_pipeline.sh"]
