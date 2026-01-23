# BERT Sentiment Analysis - Windows PowerShell Runner
# Run this with: .\run_windows.ps1

function Show-Menu {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "BERT Sentiment Analysis - Windows Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[1] Build Docker Image" -ForegroundColor Yellow
    Write-Host "[2] Run Complete Training Pipeline" -ForegroundColor Yellow
    Write-Host "[3] Run Training (Quick - if already built)" -ForegroundColor Yellow
    Write-Host "[4] View MLflow UI" -ForegroundColor Yellow
    Write-Host "[5] Open Container Shell" -ForegroundColor Yellow
    Write-Host "[6] View Training Logs" -ForegroundColor Yellow
    Write-Host "[7] Stop All Containers" -ForegroundColor Yellow
    Write-Host "[8] Clean Up Everything" -ForegroundColor Yellow
    Write-Host "[9] Test Docker Setup" -ForegroundColor Yellow
    Write-Host "[0] Exit" -ForegroundColor Yellow
    Write-Host ""
}

function Build-Image {
    Write-Host "`n🏗️ Building Docker image..." -ForegroundColor Cyan
    Write-Host "This will take 3-5 minutes...`n" -ForegroundColor Gray
    
    docker build -t bert-sentiment:latest .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Build completed successfully!`n" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Build failed!`n" -ForegroundColor Red
    }
    
    Pause
}

function Run-Training {
    param([bool]$BuildFirst = $false)
    
    if ($BuildFirst) {
        Build-Image
    }
    
    Write-Host "`n🚀 Running training pipeline..." -ForegroundColor Cyan
    Write-Host "This will take 30-40 minutes on CPU...`n" -ForegroundColor Gray
    
    $currentPath = Get-Location
    
    docker run --rm `
        -v "${currentPath}\data:/app/data" `
        -v "${currentPath}\models:/app/models" `
        -v "${currentPath}\mlruns:/app/mlruns" `
        -v "${currentPath}\logs:/app/logs" `
        --name bert-training `
        bert-sentiment:latest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Pipeline completed successfully!`n" -ForegroundColor Green
        Write-Host "📍 Model saved to: models\checkpoints\best_model.pth" -ForegroundColor Yellow
    } else {
        Write-Host "`n❌ Training failed!`n" -ForegroundColor Red
    }
    
    Pause
}

function Start-MLflowUI {
    Write-Host "`n📊 Starting MLflow UI..." -ForegroundColor Cyan
    Write-Host "🌐 Access at: http://localhost:5000" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray
    
    $currentPath = Get-Location
    
    docker run --rm `
        -p 5000:5000 `
        -v "${currentPath}\mlruns:/app/mlruns" `
        bert-sentiment:latest `
        mlflow ui --host 0.0.0.0 --backend-store-uri file:///app/mlruns
}

function Open-Shell {
    Write-Host "`n🐚 Opening container shell..." -ForegroundColor Cyan
    Write-Host "Type 'exit' to return`n" -ForegroundColor Gray
    
    $currentPath = Get-Location
    
    docker run -it --rm `
        -v "${currentPath}\data:/app/data" `
        -v "${currentPath}\models:/app/models" `
        -v "${currentPath}\mlruns:/app/mlruns" `
        bert-sentiment:latest `
        /bin/bash
}

function View-Logs {
    Write-Host "`n📝 Training Logs:`n" -ForegroundColor Cyan
    
    if (Test-Path "logs\training.log") {
        Get-Content "logs\training.log" -Tail 50
    } else {
        Write-Host "No logs found yet. Run training first." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Pause
}

function Stop-Containers {
    Write-Host "`n🛑 Stopping all containers..." -ForegroundColor Cyan
    
    docker stop bert-training 2>$null
    docker stop mlflow-ui 2>$null
    
    Write-Host "✅ All containers stopped!`n" -ForegroundColor Green
    Pause
}

function Clean-All {
    Write-Host "`n⚠️ WARNING: This will remove Docker images and containers!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    
    if ($confirm -eq "yes") {
        Write-Host "`n🧹 Cleaning up..." -ForegroundColor Cyan
        
        docker stop bert-training 2>$null
        docker rm bert-training 2>$null
        docker rmi bert-sentiment:latest 2>$null
        
        Write-Host "✅ Cleanup completed!`n" -ForegroundColor Green
    } else {
        Write-Host "❌ Cleanup cancelled.`n" -ForegroundColor Yellow
    }
    
    Pause
}

function Test-DockerSetup {
    Write-Host "`n🧪 Testing Docker setup...`n" -ForegroundColor Cyan
    
    Write-Host "1️⃣ Checking Docker..." -ForegroundColor Yellow
    docker --version
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker is not installed!" -ForegroundColor Red
        Write-Host "📥 Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        Pause
        return
    }
    Write-Host "✅ Docker found!`n" -ForegroundColor Green
    
    Write-Host "2️⃣ Checking Docker daemon..." -ForegroundColor Yellow
    docker ps | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker daemon is not running!" -ForegroundColor Red
        Write-Host "🔧 Please start Docker Desktop" -ForegroundColor Yellow
        Pause
        return
    }
    Write-Host "✅ Docker daemon is running!`n" -ForegroundColor Green
    
    Write-Host "3️⃣ Checking disk space..." -ForegroundColor Yellow
    docker system df
    Write-Host ""
    
    Write-Host "✅ All checks passed!`n" -ForegroundColor Green
    Pause
}

# Main loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice (0-9)"
    
    switch ($choice) {
        "1" { Build-Image }
        "2" { Run-Training -BuildFirst $true }
        "3" { Run-Training -BuildFirst $false }
        "4" { Start-MLflowUI }
        "5" { Open-Shell }
        "6" { View-Logs }
        "7" { Stop-Containers }
        "8" { Clean-All }
        "9" { Test-DockerSetup }
        "0" { 
            Write-Host "`nExiting...`n" -ForegroundColor Cyan
            exit 
        }
        default { 
            Write-Host "`n❌ Invalid choice! Please try again.`n" -ForegroundColor Red
            Pause
        }
    }
}