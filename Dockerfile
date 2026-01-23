# ================================
# BERT Sentiment Analysis – Docker
# Windows-safe, CPU training
# ================================

FROM python:3.10-slim

# ----------------
# System setup
# ----------------
WORKDIR /app

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
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo \"========================================\"\n\
echo \"BERT Sentiment Analysis – Docker Pipeline\"\n\
echo \"========================================\"\n\
\n\
python - <<EOF\n\
import torch\n\
print(\"CUDA available:\", torch.cuda.is_available())\n\
EOF\n\
\n\
if [ ! -f data/processed/train.csv ]; then\n\
  echo \"📥 Preparing data...\"\n\
  python src/data/prepare_data.py\n\
else\n\
  echo \"✅ Data already prepared\"\n\
fi\n\
\n\
echo \"🚀 Training model...\"\n\
python src/models/train.py\n\
\n\
echo \"========================================\"\n\
echo \"🎉 Pipeline completed successfully\"\n\
echo \"Model location: models/checkpoints\"\n\
echo \"MLflow runs: mlruns/\"\n\
echo \"========================================\"\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# ----------------
# Default command
# ----------------
CMD ["/app/run_pipeline.sh"]

# ----------------
# Volumes (Windows-safe)
# ----------------
VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/logs"]
