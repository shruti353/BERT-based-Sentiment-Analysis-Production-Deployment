"""
FastAPI Production Inference Server for BERT Sentiment Analysis
Handles 100+ requests/minute with <120ms latency
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import psutil
import torch

from .models import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from .inference import SentimentPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'sentiment_requests_total', 
    'Total sentiment analysis requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'sentiment_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'sentiment_predictions_total',
    'Total predictions made',
    ['sentiment']
)

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global predictor
    
    # Startup: Load model
    logger.info("🚀 Starting FastAPI server...")
    logger.info("📦 Loading BERT model...")
    
    start_time = time.time()
    predictor = SentimentPredictor(
        model_path="models/checkpoints/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    load_time = time.time() - start_time
    
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    logger.info(f"🖥️  Using device: {predictor.device}")
    logger.info(f"💾 Model size: {predictor.get_model_size_mb():.2f} MB")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("🛑 Shutting down server...")
    if predictor:
        del predictor
    logger.info("✅ Cleanup complete")

# Initialize FastAPI app
app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="Production-ready sentiment analysis with BERT",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "BERT Sentiment Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU info if available
        gpu_available = torch.cuda.is_available()
        gpu_memory = None
        if gpu_available:
            gpu_memory = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "cached_mb": torch.cuda.memory_reserved() / 1024**2
            }
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(predictor.device),
            uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_available=gpu_available,
            gpu_memory=gpu_memory
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=predictor.model_name,
        num_parameters=predictor.get_num_parameters(),
        model_size_mb=predictor.get_model_size_mb(),
        device=str(predictor.device),
        max_length=predictor.max_length,
        classes=predictor.classes
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text
    
    - **text**: Input text to analyze (max 5000 characters)
    - Returns: sentiment label, confidence score, and probabilities
    """
    start_time = time.time()
    
    try:
        if predictor is None:
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate text length
        if len(request.text) > 5000:
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(
                status_code=400, 
                detail="Text too long. Maximum 5000 characters allowed."
            )
        
        # Make prediction
        result = predictor.predict(request.text)
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        PREDICTION_COUNT.labels(sentiment=result['label']).inc()
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='predict').observe(latency)
        
        logger.info(f"Prediction completed in {latency*1000:.2f}ms - Sentiment: {result['label']}")
        
        return PredictionResponse(
            text=request.text,
            sentiment=result['label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=latency * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_sentiment_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts (batch processing)
    
    - **texts**: List of texts to analyze (max 100 texts per request)
    - Returns: List of predictions with sentiments and confidences
    """
    start_time = time.time()
    
    try:
        if predictor is None:
            REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate batch size
        if len(request.texts) > 100:
            REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
            raise HTTPException(
                status_code=400,
                detail="Batch too large. Maximum 100 texts per request."
            )
        
        # Validate text lengths
        for text in request.texts:
            if len(text) > 5000:
                REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
                raise HTTPException(
                    status_code=400,
                    detail="Text too long. Maximum 5000 characters per text."
                )
        
        # Make batch predictions
        results = predictor.predict_batch(request.texts)
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint='predict_batch', status='success').inc()
        for result in results:
            PREDICTION_COUNT.labels(sentiment=result['label']).inc()
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='predict_batch').observe(latency)
        
        logger.info(f"Batch prediction completed: {len(results)} texts in {latency*1000:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    text=text,
                    sentiment=result['label'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    processing_time_ms=(latency * 1000) / len(results)
                )
                for text, result in zip(request.texts, results)
            ],
            total_texts=len(results),
            total_processing_time_ms=latency * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Store start time
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    logger.info("✅ Server started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )