"""
Optimized inference engine for BERT sentiment analysis
Targets <120ms latency for single predictions
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from typing import List, Dict
import time
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.bert_classifier import BERTSentimentClassifier

logger = logging.getLogger(__name__)

class SentimentPredictor:
    """
    Fast inference engine for sentiment predictions
    Optimized for low latency (<120ms) and high throughput (100+ req/min)
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        max_length: int = 512
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            max_length: Maximum sequence length
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_length = max_length
        self.model_name = "bert-base-uncased"
        self.classes = ["negative", "positive"]
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Warm up model (first inference is always slower)
        self._warmup()
        
        logger.info("Predictor initialized successfully")
    
    def _load_model(self, model_path: str) -> BERTSentimentClassifier:
        """Load model from checkpoint"""
        model = BERTSentimentClassifier(
            n_classes=len(self.classes),
            pretrained_model=self.model_name
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _warmup(self, n_iterations: int = 3):
        """Warm up model with dummy predictions"""
        logger.info("Warming up model...")
        dummy_text = "This is a warmup text for model initialization."
        
        for _ in range(n_iterations):
            with torch.no_grad():
                self.predict(dummy_text)
        
        logger.info("Warmup complete")
    
    def _preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def _postprocess(self, logits: torch.Tensor) -> Dict[str, any]:
        """
        Convert model output to prediction result
        
        Args:
            logits: Model output logits
        
        Returns:
            Dictionary with label, confidence, and probabilities
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        # Get prediction
        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = probs_np[pred_idx]
        label = self.classes[pred_idx]
        
        # Create probability distribution
        probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(self.classes, probs_np)
        }
        
        return {
            'label': label,
            'confidence': float(confidence),
            'probabilities': probabilities
        }
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        inputs = self._preprocess(text)
        
        # Inference
        logits = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Postprocess
        result = self._postprocess(logits)
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict sentiment for multiple texts (batch processing)
        More efficient for multiple predictions
        
        Args:
            texts: List of text strings
        
        Returns:
            List of prediction results
        """
        # Batch tokenization
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Batch inference
        logits = self.model(input_ids, attention_mask)
        
        # Postprocess batch
        results = []
        for i in range(len(texts)):
            result = self._postprocess(logits[i:i+1])
            results.append(result)
        
        return results
    
    def get_num_parameters(self) -> int:
        """Get number of model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def benchmark(self, text: str = None, n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance
        
        Args:
            text: Text to use for benchmarking (uses default if None)
            n_iterations: Number of iterations to run
        
        Returns:
            Dictionary with benchmark statistics
        """
        if text is None:
            text = "This is a sample text for benchmarking the inference speed of our model."
        
        logger.info(f"Running benchmark with {n_iterations} iterations...")
        
        latencies = []
        for _ in range(n_iterations):
            start_time = time.time()
            self.predict(text)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        stats = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_req_per_sec': 1000 / np.mean(latencies)
        }
        
        logger.info(f"Benchmark results: Mean latency = {stats['mean_latency_ms']:.2f}ms")
        logger.info(f"Throughput: {stats['throughput_req_per_sec']:.2f} req/s")
        
        return stats