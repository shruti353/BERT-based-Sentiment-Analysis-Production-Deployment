@echo off
REM BERT Sentiment Analysis - Windows Runner Script
REM Run this script to execute the complete pipeline

echo ========================================
echo BERT Sentiment Analysis - Windows Setup
echo ========================================
echo.

:MENU
echo Choose an option:
echo.
echo [1] Build Docker Image
echo [2] Run Complete Training Pipeline
echo [3] Run Training (if already built)
echo [4] View MLflow UI
echo [5] Open Container Shell
echo [6] View Training Logs
echo [7] Stop All Containers
echo [8] Clean Up Everything
echo [9] Test Docker Setup
echo [0] Exit
echo.

set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto BUILD
if "%choice%"=="2" goto BUILD_AND_RUN
if "%choice%"=="3" goto RUN
if "%choice%"=="4" goto MLFLOW
if "%choice%"=="5" goto SHELL
if "%choice%"=="6" goto LOGS
if "%choice%"=="7" goto STOP
if "%choice%"=="8" goto CLEAN
if "%choice%"=="9" goto TEST
if "%choice%"=="0" goto END

echo Invalid choice! Please try again.
echo.
goto MENU

:BUILD
echo.
echo Building Docker image...
echo This will take 3-5 minutes...
echo.
docker build -t bert-sentiment:latest .
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MENU
)
echo.
echo Build completed successfully!
echo.
pause
goto MENU

:BUILD_AND_RUN
echo.
echo Building and running complete pipeline...
echo.
docker build -t bert-sentiment:latest .
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MENU
)
echo.
echo Starting training pipeline...
echo This will take 30-40 minutes...
echo.
docker run --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\mlruns":/app/mlruns ^
    -v "%cd%\logs":/app/logs ^
    --name bert-training ^
    bert-sentiment:latest
echo.
echo Pipeline completed!
echo.
pause
goto MENU

:RUN
echo.
echo Running training pipeline...
echo This will take 30-40 minutes...
echo.
docker run --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\mlruns":/app/mlruns ^
    -v "%cd%\logs":/app/logs ^
    --name bert-training ^
    bert-sentiment:latest
echo.
echo Pipeline completed!
echo.
pause
goto MENU

:MLFLOW
echo.
echo Starting MLflow UI...
echo Access at: http://localhost:5000
echo Press Ctrl+C to stop
echo.
docker run --rm ^
    -p 5000:5000 ^
    -v "%cd%\mlruns":/app/mlruns ^
    bert-sentiment:latest ^
    mlflow ui --host 0.0.0.0 --backend-store-uri file:///app/mlruns
goto MENU

:SHELL
echo.
echo Opening container shell...
echo Type 'exit' to return
echo.
docker run -it --rm ^
    -v "%cd%\data":/app/data ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\mlruns":/app/mlruns ^
    bert-sentiment:latest ^
    /bin/bash
goto MENU

:LOGS
echo.
echo Viewing training logs...
echo.
if exist "logs\training.log" (
    type logs\training.log
) else (
    echo No logs found yet. Run training first.
)
echo.
pause
goto MENU

:STOP
echo.
echo Stopping all containers...
docker stop bert-training 2>nul
docker stop mlflow-ui 2>nul
echo.
echo All containers stopped!
echo.
pause
goto MENU

:CLEAN
echo.
echo WARNING: This will remove all Docker images and containers!
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    echo.
    echo Cleaning up...
    docker stop bert-training 2>nul
    docker rm bert-training 2>nul
    docker rmi bert-sentiment:latest 2>nul
    echo.
    echo Cleanup completed!
) else (
    echo.
    echo Cleanup cancelled.
)
echo.
pause
goto MENU

:TEST
echo.
echo Testing Docker setup...
echo.
echo 1. Checking Docker...
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    goto MENU
)
echo    Docker found!
echo.
echo 2. Checking Docker daemon...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker daemon is not running!
    echo Please start Docker Desktop
    pause
    goto MENU
)
echo    Docker daemon is running!
echo.
echo 3. Checking disk space...
echo.
docker system df
echo.
echo All checks passed!
echo.
pause
goto MENU

:END
echo.
echo Exiting...
echo.
exit /b 0