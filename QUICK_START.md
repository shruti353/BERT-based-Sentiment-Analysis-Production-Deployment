#  Quick Start Guide - BERT Sentiment Analysis

## Step-by-Step Setup (15 minutes to first results!)

### 1️⃣ Environment Setup (3 minutes)

```bash
# Clone/create your project directory
mkdir bert-sentiment-mlops
cd bert-sentiment-mlops

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Project Structure Setup (1 minute)

```bash
# Make setup script executable (Mac/Linux)
chmod +x setup_project.sh

# Run setup script
./setup_project.sh

# Or manually create directories (Windows/all platforms):
mkdir -p data/raw data/processed src/data src/models models/checkpoints mlruns logs
touch src/__init__.py src/data/__init__.py src/models/__init__.py
```

### 3️⃣ Data Preparation (5 minutes)

```bash
# Download and preprocess IMDB dataset
python src/data/prepare_data.py
```

**What this does:**
- Downloads 50,000 IMDB movie reviews
- Cleans and preprocesses text
- Creates train/validation/test splits (70/15/15)
- Saves processed data to `data/processed/`
- Generates dataset statistics

**Expected output:**
```
📥 Downloading IMDB dataset...
✅ Downloaded 25,000 training samples
✅ Downloaded 25,000 test samples

🔧 Preprocessing train dataset...
📊 Creating data splits...
  Train: 21,250 samples
  Val:   3,750 samples
  Test:  25,000 samples

💾 Saving processed datasets...
✅ Saved to data/processed/

📈 DATASET STATISTICS
==================================================
Train samples: 21,250
Val samples:   3,750
Test samples:  25,000
Number of classes: 2
Class distribution (train):
  Positive (1): 10,625 (50.0%)
  Negative (0): 10,625 (50.0%)
==================================================
```

### 4️⃣ Start Training (5 minutes setup, then let it run)

```bash
# Start training with MLflow tracking
python src/models/train.py
```

**What this does:**
- Loads pre-trained BERT-base-uncased model
- Fine-tunes on IMDB dataset
- Tracks all metrics with MLflow
- Saves best model automatically
- Evaluates on test set

**Training progress:**
```
🖥️  Using device: cuda (or cpu)
📊 Model parameters: 109,483,778
💾 Model size: 417.65 MB

🚀 Starting training for 4 epochs...

Epoch 1/4
==================================================
Training: 100%|████████████| 1328/1328 [08:32<00:00]
loss: 0.3521, acc: 0.8456

Evaluating: 100%|████████████| 235/235 [01:15<00:00]

Train Loss: 0.3521 | Train Acc: 0.8456
Val Loss:   0.2845 | Val Acc:   0.8834 | Val F1: 0.8829
✅ New best model saved! (Acc: 0.8834, F1: 0.8829)
```

### 5️⃣ View Results in MLflow UI

```bash
# In a new terminal (keep training running)
mlflow ui
```

Then open browser to: **http://localhost:5000**

You'll see:
- All experiment runs
- Real-time metrics (loss, accuracy, F1)
- Model parameters
- Training curves
- Saved model artifacts

---

## 📊 Expected Results (After ~30-40 min training on GPU)

### **Quick Wins You'll Achieve:**

✅ **Accuracy**: ~88-92% on validation set (will reach 94%+ with hyperparameter tuning)  
✅ **F1 Score**: ~0.88-0.92  
✅ **Model Size**: ~418 MB (BERT-base)  
✅ **Training Time**: 
- GPU (Tesla T4): ~8-10 min/epoch
- CPU: ~40-50 min/epoch

### **What Gets Saved:**

```
bert-sentiment-mlops/
├── data/processed/          # Cleaned datasets
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── dataset_stats.json
├── models/checkpoints/      # Model checkpoints
│   └── best_model.pth      # Best performing model
├── mlruns/                  # MLflow tracking data
│   └── [experiment_id]/
│       ├── metrics/
│       ├── params/
│       └── artifacts/
```

---

## 🎯 Next Steps After First Training

### **Immediate Next Steps:**

1. **View MLflow Dashboard**
   ```bash
   mlflow ui
   ```
   - Compare different runs
   - View training curves
   - Check confusion matrix

2. **Test Inference**
   ```python
   # Quick test script
   from transformers import BertTokenizer
   import torch
   from src.models.bert_classifier import BERTSentimentClassifier
   
   # Load model
   model = BERTSentimentClassifier()
   checkpoint = torch.load('models/checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   
   # Test prediction
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   text = "This movie was absolutely amazing! I loved every minute."
   
   encoding = tokenizer.encode_plus(
       text, max_length=512, padding='max_length',
       truncation=True, return_tensors='pt'
   )
   
   with torch.no_grad():
       outputs = model(encoding['input_ids'], encoding['attention_mask'])
       _, pred = torch.max(outputs, dim=1)
   
   print(f"Sentiment: {'Positive' if pred.item() == 1 else 'Negative'}")
   ```

3. **Experiment with Hyperparameters**
   - Modify `config` in `train.py`:
     - Try learning rates: `1e-5`, `3e-5`, `5e-5`
     - Try batch sizes: `8`, `16`, `32`
     - Try epochs: `3`, `4`, `5`
   - All experiments tracked in MLflow!

---

## 🔧 Troubleshooting

### **GPU Memory Issues:**
```python
# In train.py, reduce batch_size
'batch_size': 8,  # Instead of 16
```

### **Slow Training on CPU:**
```python
# Use smaller max_length
'max_length': 256,  # Instead of 512
# Or use DistilBERT (faster, smaller)
'model_name': 'distilbert-base-uncased',
```

### **Data Download Issues:**
```python
# Manual download if needed
from datasets import load_dataset
dataset = load_dataset("imdb", cache_dir="./data/raw")
```

---

## 📈 Performance Benchmarks

| Configuration | Accuracy | F1 Score | Training Time/Epoch |
|--------------|----------|----------|---------------------|
| BERT-base (GPU) | 91-93% | 0.91-0.93 | ~8-10 min |
| BERT-base (CPU) | 91-93% | 0.91-0.93 | ~45 min |
| DistilBERT (GPU) | 89-91% | 0.89-0.91 | ~5 min |

---

## ✅ Success Checklist

After running the pipeline, you should have:

- [ ] Downloaded and preprocessed 50,000 reviews
- [ ] Created train/val/test splits
- [ ] Fine-tuned BERT model for 4 epochs
- [ ] Achieved >88% validation accuracy
- [ ] Saved best model checkpoint
- [ ] Tracked all metrics in MLflow
- [ ] Generated classification report on test set

**You're now ready to move to Phase 2: Building the FastAPI inference server!** 🚀