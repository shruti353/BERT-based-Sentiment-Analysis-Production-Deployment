# 🐳 Docker Setup Guide - BERT Sentiment Analysis

## Complete Automated Pipeline in Docker

This guide shows you how to run the **entire training pipeline** (data download → preprocessing → training → MLflow tracking) inside Docker containers.

---

## 🎯 Quick Start (3 Commands!)

```bash
# 1. Test your Docker setup
make test

# 2. Build the Docker image
make build

# 3. Run the complete pipeline!
make train
```

**That's it!** Docker will automatically:
- ✅ Download IMDB dataset (50,000 reviews)
- ✅ Preprocess and clean data
- ✅ Fine-tune BERT model
- ✅ Track everything in MLflow
- ✅ Save trained model

---

## 📋 Prerequisites

### Required:
- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **8GB RAM** minimum (16GB recommended)
- **20GB disk space**

### Optional (for GPU training):
- **NVIDIA GPU** with CUDA support
- **nvidia-docker2** installed

### Installation:

**Docker (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**Docker Compose:**
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**NVIDIA Docker (for GPU):**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## 🚀 Usage Options

### **Option 1: Using Make Commands (Recommended)**

```bash
# View all available commands
make help

# Build and train (CPU)
make train

# Build and train (GPU)
make train-gpu

# View MLflow UI
make mlflow-ui  # Access at http://localhost:5000

# View training logs
make logs

# Stop all containers
make stop

# Clean up everything
make clean
```

### **Option 2: Using Docker Compose**

```bash
# Start complete stack (training + MLflow UI)
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f training

# Stop services
docker-compose down
```

### **Option 3: Direct Docker Commands**

```bash
# Build image
docker build -t bert-sentiment-training:latest .

# Run training pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/logs:/app/logs \
  bert-sentiment-training:latest

# Run with GPU
docker run --gpus all --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  bert-sentiment-training:gpu
```

---

## 📁 Volume Mounts Explained

Docker containers are ephemeral, so we use volumes to persist data:

```yaml
volumes:
  - ./data:/app/data           # Dataset storage
  - ./models:/app/models       # Trained models
  - ./mlruns:/app/mlruns       # MLflow experiments
  - ./logs:/app/logs           # Training logs
```

**After training, you'll find:**
```
your-project/
├── data/
│   ├── raw/              # Downloaded IMDB dataset
│   └── processed/        # Preprocessed CSV files
├── models/
│   └── checkpoints/
│       └── best_model.pth   # ⭐ Your trained model!
├── mlruns/               # MLflow tracking data
└── logs/                 # Training logs
```

---

## 🎮 Interactive Usage

### Open Shell Inside Container

```bash
# Option 1: Using Make
make shell

# Option 2: Direct command
docker run -it --rm \
  -v $(pwd):/app \
  bert-sentiment-training:latest \
  /bin/bash
```

**Inside the container you can:**
```bash
# Run data preparation only
python src/data/prepare_data.py

# Run training only
python src/models/train.py

# Start MLflow UI
mlflow ui --host 0.0.0.0

# Test model inference
python -c "from src.models.bert_classifier import BERTSentimentClassifier; print('Model loaded!')"
```

---

## ⚙️ Configuration

### CPU vs GPU Training

**CPU Training (Default):**
- Uses `Dockerfile`
- Training time: ~40-50 min/epoch
- Memory: 4-6 GB RAM

**GPU Training:**
- Uses `Dockerfile.gpu`
- Training time: ~8-10 min/epoch
- Requires NVIDIA GPU

```bash
# GPU training
make train-gpu

# Or with docker-compose (uncomment GPU section)
docker-compose up training
```

### Adjust Resources

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'      # Adjust CPU cores
      memory: 8G     # Adjust memory
```

### Modify Training Parameters

**Option 1: Environment Variables**
```bash
docker run --rm \
  -e EPOCHS=5 \
  -e BATCH_SIZE=32 \
  -e LEARNING_RATE=3e-5 \
  -v $(pwd)/data:/app/data \
  bert-sentiment-training:latest
```

**Option 2: Edit config in code**
Modify `src/models/train.py` before building image.

---

## 📊 Monitoring Training

### View Live Logs

```bash
# Using Make
make logs

# Using Docker Compose
docker-compose logs -f training

# Direct Docker
docker logs -f bert-training
```

### MLflow UI

```bash
# Start MLflow UI
make mlflow-ui

# Or with Docker Compose
docker-compose up mlflow-ui

# Access at: http://localhost:5000
```

### GPU Monitoring (if available)

```bash
# Watch GPU usage
make gpu-monitor

# Or directly
watch -n 1 nvidia-smi
```

---

## 🔧 Troubleshooting

### Issue: "Out of Memory" Error

**Solution 1: Reduce batch size**
```bash
# Edit src/models/train.py before building
'batch_size': 8,  # Reduced from 16
```

**Solution 2: Increase Docker memory**
```bash
# Docker Desktop → Settings → Resources → Memory
# Increase to 8GB or more
```

### Issue: Training is slow on CPU

**Solutions:**
- Use GPU version: `make train-gpu`
- Reduce epochs for testing: `'epochs': 2`
- Use DistilBERT (faster): `'model_name': 'distilbert-base-uncased'`

### Issue: "No space left on device"

```bash
# Clean up Docker
docker system prune -a

# Check disk usage
make disk-usage

# Remove old images
docker image prune -a
```

### Issue: Can't connect to MLflow UI

```bash
# Ensure MLflow container is running
docker-compose ps

# Check logs
docker-compose logs mlflow-ui

# Restart service
docker-compose restart mlflow-ui
```

### Issue: GPU not detected

```bash
# Test GPU availability
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker GPU runtime
docker run --rm --gpus all ubuntu nvidia-smi
```

---

## 📈 Expected Output

### During Build:
```
🏗️  Building Docker image...
[+] Building 245.3s (18/18) FINISHED
 => [internal] load build definition
 => [internal] load .dockerignore
 => [base 1/5] FROM python:3.10-slim
 => [base 2/5] RUN apt-get update && apt-get install...
 => [base 3/5] COPY requirements.txt .
 => [base 4/5] RUN pip install --no-cache-dir...
✅ Build complete!
```

### During Training:
```
========================================
BERT Sentiment Analysis Training Pipeline
========================================

📥 Step 1/3: Downloading and preprocessing data...
✅ Downloaded 25,000 training samples
✅ Downloaded 25,000 test samples
✅ Data preparation complete!

🚀 Step 2/3: Starting model training...
🖥️  Using device: cuda
📊 Model parameters: 109,483,778

Epoch 1/4
==================================================
Training: 100%|████████| 1328/1328 [08:32<00:00]
Val Acc: 0.8834 | Val F1: 0.8829
✅ New best model saved!

🎉 Pipeline completed successfully!
========================================
✅ Model saved: models/checkpoints/best_model.pth
📦 Model size: 418M
📈 MLflow runs: 1
```

---

## 🎯 Advanced Usage

### Multi-Stage Training Pipeline

```bash
# Stage 1: Data preparation only
docker run --rm \
  -v $(pwd)/data:/app/data \
  bert-sentiment-training:latest \
  python src/data/prepare_data.py

# Stage 2: Training with prepared data
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  bert-sentiment-training:latest \
  python src/models/train.py
```

### Experiment Tracking with Multiple Runs

```bash
# Run experiment 1
docker run --rm \
  -e LEARNING_RATE=1e-5 \
  -v $(pwd):/app \
  bert-sentiment-training:latest

# Run experiment 2
docker run --rm \
  -e LEARNING_RATE=3e-5 \
  -v $(pwd):/app \
  bert-sentiment-training:latest

# Compare in MLflow UI
make mlflow-ui
```

### Continuous Training (Auto-restart)

```yaml
# In docker-compose.yml
services:
  training:
    restart: always  # or on-failure
```

---

## 📦 Production Deployment

### Build for Production

```bash
# Build optimized image
docker build --target training \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t bert-sentiment-training:prod .

# Push to registry
docker tag bert-sentiment-training:prod your-registry/bert-sentiment:latest
docker push your-registry/bert-sentiment:latest
```

### Deploy to Cloud

**AWS ECS:**
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin your-ecr-url
docker tag bert-sentiment-training:latest your-ecr-url/bert-sentiment:latest
docker push your-ecr-url/bert-sentiment:latest
```

**Google Cloud Run:**
```bash
# Push to GCR
docker tag bert-sentiment-training:latest gcr.io/your-project/bert-sentiment:latest
docker push gcr.io/your-project/bert-sentiment:latest
```

---

## 🔐 Security Best Practices

1. **Don't include sensitive data in image**
   - Use volumes for data
   - Use environment variables for secrets

2. **Use specific Python version**
   ```dockerfile
   FROM python:3.10.12-slim  # Not just 3.10
   ```

3. **Run as non-root user**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

4. **Scan for vulnerabilities**
   ```bash
   docker scan bert-sentiment-training:latest
   ```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [MLflow Docker](https://www.mlflow.org/docs/latest/docker.html)

---

## ✅ Verification Checklist

After running `make train`, verify:

- [ ] Data downloaded: `ls data/processed/train.csv`
- [ ] Model trained: `ls models/checkpoints/best_model.pth`
- [ ] MLflow tracking: `ls mlruns/`
- [ ] Logs created: `ls logs/`
- [ ] Can access MLflow UI: http://localhost:5000

**If all checked → You're ready for production! 🚀**