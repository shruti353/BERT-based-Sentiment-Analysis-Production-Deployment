"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    """Single text prediction request"""
    text: str = Field(
        ...,
        description="Text to analyze for sentiment",
        min_length=1,
        max_length=5000,
        example="This movie was absolutely fantastic! I loved every minute of it."
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()

class PredictionResponse(BaseModel):
    """Single prediction response"""
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment label", example="positive")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1, example=0.95)
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution over all classes",
        example={"negative": 0.05, "positive": 0.95}
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", example=87.5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic!",
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "negative": 0.05,
                    "positive": 0.95
                },
                "processing_time_ms": 87.5
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100,
        example=[
            "This is amazing!",
            "I hate this product.",
            "It's okay, nothing special."
        ]
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('texts list cannot be empty')
        
        for text in v:
            if not text.strip():
                raise ValueError('texts cannot contain empty strings')
            if len(text) > 5000:
                raise ValueError('Each text must be less than 5000 characters')
        
        return [text.strip() for text in v]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_texts: int = Field(..., description="Total number of texts processed")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "text": "This is amazing!",
                        "sentiment": "positive",
                        "confidence": 0.98,
                        "probabilities": {"negative": 0.02, "positive": 0.98},
                        "processing_time_ms": 45.2
                    },
                    {
                        "text": "I hate this product.",
                        "sentiment": "negative",
                        "confidence": 0.94,
                        "probabilities": {"negative": 0.94, "positive": 0.06},
                        "processing_time_ms": 45.2
                    }
                ],
                "total_texts": 2,
                "total_processing_time_ms": 90.4
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cuda")
    uptime_seconds: float = Field(..., example=3600.5)
    cpu_percent: float = Field(..., example=45.2)
    memory_percent: float = Field(..., example=62.8)
    gpu_available: bool = Field(..., example=True)
    gpu_memory: Optional[Dict[str, float]] = Field(
        None, 
        example={"allocated_mb": 512.5, "cached_mb": 1024.0}
    )

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., example="bert-base-uncased")
    num_parameters: int = Field(..., example=109483778)
    model_size_mb: float = Field(..., example=417.65)
    device: str = Field(..., example="cuda")
    max_length: int = Field(..., example=512)
    classes: List[str] = Field(..., example=["negative", "positive"])

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Model not loaded",
                "status_code": 503
            }
        }