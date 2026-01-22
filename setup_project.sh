#!/bin/bash

# BERT Sentiment Analysis Project Setup Script
echo "🚀 Setting up BERT Sentiment Analysis Project..."

# Create directory structure
mkdir -p data/{raw,processed}
mkdir -p src/{data,models,utils}
mkdir -p notebooks
mkdir -p models/checkpoints
mkdir -p mlruns
mkdir -p logs
mkdir -p configs

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/checkpoints/*
!models/checkpoints/.gitkeep
*.pth
*.bin
*.onnx

# MLflow
mlruns/
mlartifacts/

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/

# Logs
logs/
*.log

# Environment
.env
EOF

# Create placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/checkpoints/.gitkeep

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory structure:"
tree -L 2 -I '__pycache__|*.pyc' || ls -R

echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run data preparation: python src/data/prepare_data.py"
echo "3. Start training: python src/models/train.py"