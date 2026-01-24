FROM python:3.10

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Env vars
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Project structure
RUN mkdir -p data/raw data/processed models/checkpoints mlruns logs cache src/data src/models src/utils

# Copy code
COPY src/ ./src/

# Init files
RUN touch src/__init__.py src/data/__init__.py src/models/__init__.py src/utils/__init__.py

# Create pipeline script (PROPER WAY)
RUN printf '#!/bin/bash\n\
set -e\n\
echo "========================================"\n\
echo "BERT Sentiment Analysis – Docker Pipeline"\n\
echo "========================================"\n\
python - <<EOF\n\
import torch\n\
print("CUDA available:", torch.cuda.is_available())\n\
EOF\n\
if [ ! -f data/processed/train.csv ]; then\n\
  echo "📥 Preparing data..."\n\
  python src/data/prepare_data.py\n\
else\n\
  echo "✅ Data already prepared"\n\
fi\n\
echo "🚀 Training model..."\n\
python src/models/train.py\n\
echo "🎉 Pipeline completed!"\n' > /app/run_pipeline.sh \
&& sed -i 's/\r$//' /app/run_pipeline.sh \
&& chmod +x /app/run_pipeline.sh

CMD ["/app/run_pipeline.sh"]

VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/logs", "/app/cache"]
