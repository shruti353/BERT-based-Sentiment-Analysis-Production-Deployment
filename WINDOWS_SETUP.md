# 🪟 Windows Setup Guide - BERT Sentiment Analysis

## Complete Docker Pipeline for Windows

This guide is specifically for **Windows users** to run the entire BERT training pipeline with Docker.

---

## 🎯 Quick Start (3 Steps - 5 Minutes Setup!)

### **Option A: Using PowerShell (Recommended - Modern Windows)**

```powershell
# 1. Open PowerShell as Administrator
# Right-click PowerShell → Run as Administrator

# 2. Navigate to project folder
cd path\to\bert-sentiment-mlops

# 3. Run the PowerShell script
.\run_windows.ps1
```

### **Option B: Using Command Prompt (CMD)**

```cmd
# 1. Open Command Prompt
# Press Win+R, type 'cmd', press Enter

# 2. Navigate to project folder
cd path\to\bert-sentiment-mlops

# 3. Run the batch script
run_windows.bat
```

### **Option C: Manual Docker Commands**

```cmd
# Build the image
docker build -t bert-sentiment:latest .

# Run the complete pipeline
docker run --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\mlruns":/app/mlruns ^
    -v "%cd%\logs":/app/logs ^
    bert-sentiment:latest
```

---

## 📋 Prerequisites

### **1. Install Docker Desktop for Windows**

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
2. Run the installer (Docker Desktop Installer.exe)
3. Follow installation wizard
4. **Restart your computer** after installation
5. Start Docker Desktop from Start Menu
6. Wait until Docker icon shows "Docker Desktop is running"

**System Requirements:**
- Windows 10 64-bit (Pro, Enterprise, or Education) or Windows 11
- WSL 2 feature enabled (Docker will help you enable this)
- At least 8GB RAM
- 20GB free disk space

### **2. Verify Docker Installation**

```cmd
# Check Docker version
docker --version

# Check if Docker is running
docker ps

# Expected output: Empty list (no containers running yet)
```

### **3. Enable File Sharing (Important!)**

Docker Desktop → Settings → Resources → File Sharing
- Make sure your project drive (C:, D:, etc.) is shared
- Apply & Restart

---

## 🚀 Running the Pipeline

### **Interactive Menu (Easiest)**

**Using PowerShell:**
```powershell
.\run_windows.ps1
```

**Using CMD:**
```cmd
run_windows.bat
```

**You'll see this menu:**
```
========================================
BERT Sentiment Analysis - Windows Setup
========================================

[1] Build Docker Image
[2] Run Complete Training Pipeline
[3] Run Training (if already built)
[4] View MLflow UI
[5] Open Container Shell
[6] View Training Logs
[7] Stop All Containers
[8] Clean Up Everything
[9] Test Docker Setup
[0] Exit

Enter your choice (0-9):
```

### **Recommended Workflow:**

1. **First time:** Choose `[9]` to test Docker setup
2. Then choose `[2]` to build and run everything
3. Wait 30-40 minutes (grab coffee ☕)
4. After training, choose `[4]` to view MLflow UI
5. Open browser: http://localhost:5000

---

## 📁 Windows File Paths

After running the pipeline, you'll find:

```
C:\Users\YourName\bert-sentiment-mlops\
│
├── data\
│   ├── processed\
│   │   ├── train.csv           # 21,250 training samples
│   │   ├── val.csv             # 3,750 validation samples
│   │   └── test.csv            # 25,000 test samples
│
├── models\
│   └── checkpoints\
│       └── best_model.pth      # ⭐ Your trained model! (~418 MB)
│
├── mlruns\                     # MLflow experiment tracking
│   └── 0\
│       └── [experiment_id]\
│           ├── metrics\
│           └── artifacts\
│
└── logs\                       # Training logs
```

---

## 💻 Manual Docker Commands for Windows

### **Build Image:**
```cmd
docker build -t bert-sentiment:latest .
```

### **Run Complete Pipeline:**
```cmd
docker run --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\mlruns":/app/mlruns ^
    -v "%cd%\logs":/app/logs ^
    --name bert-training ^
    bert-sentiment:latest
```

### **View MLflow UI:**
```cmd
docker run --rm ^
    -p 5000:5000 ^
    -v "%cd%\mlruns":/app/mlruns ^
    bert-sentiment:latest ^
    mlflow ui --host 0.0.0.0
```

Then open: http://localhost:5000

### **Open Shell Inside Container:**
```cmd
docker run -it --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    bert-sentiment:latest ^
    /bin/bash
```

### **Stop Container:**
```cmd
docker stop bert-training
```

### **View Running Containers:**
```cmd
docker ps
```

### **View All Containers (including stopped):**
```cmd
docker ps -a
```

---

## 🔧 Troubleshooting Windows-Specific Issues

### **Issue 1: "Docker daemon is not running"**

**Solution:**
1. Open Docker Desktop from Start Menu
2. Wait for "Docker Desktop is running" message
3. Try command again

### **Issue 2: "Drive has not been shared"**

**Solution:**
1. Open Docker Desktop
2. Settings → Resources → File Sharing
3. Add your drive (C:)
4. Click "Apply & Restart"

### **Issue 3: WSL 2 installation incomplete**

**Solution:**
```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart computer
```

### **Issue 4: "Access denied" when mounting volumes**

**Solution:**
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → Run as Administrator
.\run_windows.ps1
```

### **Issue 5: Paths with spaces**

If your path has spaces (e.g., "My Projects"), use quotes:
```cmd
docker run --rm ^
    -v "%cd%\data":/app/data ^
    ...
```

### **Issue 6: Slow performance**

**Solutions:**
1. **Increase Docker memory:**
   - Docker Desktop → Settings → Resources
   - Move Memory slider to at least 8GB
   - Click "Apply & Restart"

2. **Use WSL 2 backend (faster):**
   - Docker Desktop → Settings → General
   - Check "Use WSL 2 based engine"
   - Apply & Restart

3. **Store project in WSL filesystem:**
   ```bash
   # In PowerShell
   wsl
   cd /home/yourname/
   git clone your-repo
   ```

### **Issue 7: Cannot find PowerShell script**

**Solution:**
```powershell
# Enable script execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run
.\run_windows.ps1
```

### **Issue 8: Out of disk space**

**Solution:**
```cmd
# Clean Docker cache
docker system prune -a

# Check Docker disk usage
docker system df
```

---

## ⚡ Performance Tips for Windows

### **1. Use WSL 2 Backend**
WSL 2 is much faster than the old Hyper-V backend.

Docker Desktop → Settings → General → "Use WSL 2 based engine"

### **2. Store Files in WSL**
For best performance, keep your project in the WSL filesystem:

```powershell
# Open WSL
wsl

# Navigate to WSL home
cd ~

# Clone/copy your project here
```

### **3. Allocate More Resources**

Docker Desktop → Settings → Resources:
- **CPU:** Set to at least 4 cores
- **Memory:** Set to at least 8GB
- **Disk:** At least 20GB

### **4. Disable Antivirus for Docker Folders**

Add these to Windows Defender exclusions:
- `C:\ProgramData\Docker`
- `C:\Users\YourName\.docker`
- Your project folder

---

## 📊 Expected Performance on Windows

| Hardware | Build Time | Training Time (4 epochs) |
|----------|-----------|------------------------|
| **CPU Only** (i5/i7) | 3-5 min | 150-180 min |
| **CPU** (High-end) | 3-5 min | 100-120 min |
| **With GPU** (NVIDIA) | 4-6 min | 30-40 min |

---

## 🎮 Step-by-Step Walkthrough

### **Complete First-Time Setup:**

```cmd
# Step 1: Open Command Prompt
Win+R → type 'cmd' → Enter

# Step 2: Navigate to your project
cd C:\Users\YourName\Documents\bert-sentiment-mlops

# Step 3: Test Docker
docker --version

# Step 4: Run the menu
run_windows.bat

# Step 5: Choose [9] - Test Docker Setup
# Verify all checks pass

# Step 6: Choose [2] - Run Complete Pipeline
# This will:
#   - Build Docker image (5 min)
#   - Download data (3 min)
#   - Train model (30-40 min)
#   - Save results

# Step 7: Wait for completion
# You'll see: "Pipeline completed successfully!"

# Step 8: View results
# Choose [4] - View MLflow UI
# Open browser: http://localhost:5000

# Step 9: Check your trained model
dir models\checkpoints\best_model.pth
```

---

## 🔍 Verification Checklist

After running the pipeline, verify:

```cmd
# Check if data was downloaded
dir data\processed\train.csv

# Check if model was trained
dir models\checkpoints\best_model.pth

# Check model size (should be ~418 MB)
dir models\checkpoints\

# Check MLflow experiments
dir mlruns\

# View logs
type logs\training.log
```

**Expected output:**
```
✅ data\processed\train.csv exists
✅ models\checkpoints\best_model.pth exists (~418 MB)
✅ mlruns\0\[experiment_id]\ exists
✅ Logs show: "Training complete! Accuracy: 0.89+"
```

---

## 🌐 Access MLflow Dashboard

```cmd
# Option 1: Using menu
run_windows.bat
→ Choose [4]

# Option 2: Manual command
docker run --rm -p 5000:5000 -v "%cd%\mlruns":/app/mlruns bert-sentiment:latest mlflow ui --host 0.0.0.0

# Option 3: PowerShell
.\run_windows.ps1
→ Choose [4]
```

**Then open browser:**
```
http://localhost:5000
```

---

## 🎯 Quick Commands Reference

```cmd
REM Build image
docker build -t bert-sentiment:latest .

REM Run training
docker run --rm -v "%cd%\data":/app/data -v "%cd%\models":/app/models -v "%cd%\mlruns":/app/mlruns bert-sentiment:latest

REM View MLflow UI
docker run --rm -p 5000:5000 -v "%cd%\mlruns":/app/mlruns bert-sentiment:latest mlflow ui --host 0.0.0.0

REM Stop container
docker stop bert-training

REM View logs
docker logs bert-training

REM Clean up
docker system prune -a
```

---

## 🚀 Next Steps After Training

1. **View Results:**
   ```cmd
   run_windows.bat → [4] View MLflow UI
   ```

2. **Test Model Inference:**
   ```cmd
   run_windows.bat → [5] Open Shell
   
   # Inside container:
   python
   >>> from src.models.bert_classifier import BERTSentimentClassifier
   >>> # Test your model!
   ```

3. **Run More Experiments:**
   - Edit `src\models\train.py`
   - Change hyperparameters
   - Rebuild: `run_windows.bat → [1]`
   - Run: `run_windows.bat → [3]`

4. **Build FastAPI Server:**
   - Next phase: Production API deployment
   - I can help you build this next!

---

## 💡 Pro Tips for Windows Users

1. **Use PowerShell over CMD** - Better Unicode support and colors
2. **Pin Docker Desktop to taskbar** - Quick access
3. **Use Windows Terminal** - Modern, tabbed terminal
4. **Enable Docker Auto-start** - Docker Desktop → Settings → General
5. **Monitor resources** - Task Manager → Performance tab

---

## ✅ Success!

You should now have:

- ✅ Docker running on Windows
- ✅ Complete training pipeline automated
- ✅ Trained BERT model (~89-92% accuracy)
- ✅ MLflow experiments tracked
- ✅ All data and models saved locally

**You're now ready to deploy to production!** 🎉

Need help with:
- Building the FastAPI server?
- Model optimization?
- Cloud deployment?

Let me know! 