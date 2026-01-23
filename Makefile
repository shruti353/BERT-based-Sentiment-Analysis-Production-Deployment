# Makefile for BERT Sentiment Analysis Docker Operations

.PHONY: help build build-gpu train train-gpu mlflow-ui stop clean logs shell test

# Default target
help:
	@echo "=========================================="
	@echo "BERT Sentiment Analysis - Docker Commands"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build         - Build CPU Docker image"
	@echo "  make build-gpu     - Build GPU Docker image"
	@echo "  make train         - Run complete training pipeline (CPU)"
	@echo "  make train-gpu     - Run complete training pipeline (GPU)"
	@echo "  make mlflow-ui     - Start MLflow UI only"
	@echo "  make stop          - Stop all containers"
	@echo "  make clean         - Remove containers and images"
	@echo "  make logs          - View training logs"
	@echo "  make shell         - Open shell in training container"
	@echo "  make test          - Test Docker setup"
	@echo ""

# Build CPU image
build:
	@echo "🏗️  Building CPU Docker image..."
	docker build -t bert-sentiment-training:latest -f Dockerfile .
	@echo "✅ Build complete!"

# Build GPU image
build-gpu:
	@echo "🏗️  Building GPU Docker image..."
	docker build -t bert-sentiment-training:gpu -f Dockerfile.gpu .
	@echo "✅ GPU build complete!"

# Run training pipeline (CPU)
train: build
	@echo "🚀 Starting training pipeline (CPU)..."
	docker-compose up training
	@echo "✅ Training complete!"

# Run training pipeline (GPU)
train-gpu: build-gpu
	@echo "🚀 Starting GPU-accelerated training..."
	docker run --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/logs:/app/logs \
		--name bert-training-gpu \
		bert-sentiment-training:gpu
	@echo "✅ GPU training complete!"

# Start MLflow UI
mlflow-ui:
	@echo "📊 Starting MLflow UI..."
	@echo "🌐 Access at: http://localhost:5000"
	docker-compose up mlflow-ui

# Run complete stack (training + MLflow UI)
up: build
	@echo "🚀 Starting complete stack..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "📊 MLflow UI: http://localhost:5000"
	@echo "📝 View logs: make logs"

# Stop all containers
stop:
	@echo "🛑 Stopping containers..."
	docker-compose down
	@echo "✅ All containers stopped!"

# Clean up everything
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker rmi bert-sentiment-training:latest bert-sentiment-training:gpu 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Deep clean (including data and models)
deep-clean: clean
	@echo "⚠️  WARNING: This will delete all data and models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* models/checkpoints/* mlruns/* logs/*; \
		echo "✅ Deep clean complete!"; \
	else \
		echo "❌ Cancelled"; \
	fi

# View training logs
logs:
	@echo "📝 Viewing training logs..."
	docker-compose logs -f training

# Open shell in training container
shell:
	@echo "🐚 Opening shell in training container..."
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/mlruns:/app/mlruns \
		bert-sentiment-training:latest \
		/bin/bash

# Test Docker setup
test:
	@echo "🧪 Testing Docker setup..."
	@echo "1️⃣ Checking Docker..."
	@docker --version || (echo "❌ Docker not found!" && exit 1)
	@echo "✅ Docker found!"
	@echo ""
	@echo "2️⃣ Checking Docker Compose..."
	@docker-compose --version || (echo "❌ Docker Compose not found!" && exit 1)
	@echo "✅ Docker Compose found!"
	@echo ""
	@echo "3️⃣ Checking GPU support (optional)..."
	@nvidia-smi 2>/dev/null && echo "✅ NVIDIA GPU detected!" || echo "ℹ️  No GPU detected (CPU mode will be used)"
	@echo ""
	@echo "✅ All checks passed!"

# Data preparation only
prepare-data: build
	@echo "📥 Running data preparation only..."
	docker run --rm \
		-v $(PWD)/data:/app/data \
		bert-sentiment-training:latest \
		python src/data/prepare_data.py

# Quick training test (1 epoch)
quick-test: build
	@echo "⚡ Running quick training test (1 epoch)..."
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-e EPOCHS=1 \
		bert-sentiment-training:latest

# Monitor GPU usage (if available)
gpu-monitor:
	@echo "📊 Monitoring GPU usage..."
	@watch -n 1 nvidia-smi

# Show disk usage
disk-usage:
	@echo "💾 Disk usage:"
	@echo ""
	@du -sh data/ models/ mlruns/ 2>/dev/null || echo "No data yet"

# Backup models
backup:
	@echo "💾 Backing up models..."
	@mkdir -p backups
	@tar -czf backups/models-$(shell date +%Y%m%d-%H%M%S).tar.gz models/
	@echo "✅ Backup complete!"