"""
BERT-based Sentiment Classification Model
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTSentimentClassifier(nn.Module):
    """
    BERT model for binary sentiment classification
    """
    def __init__(self, n_classes=2, dropout=0.3, pretrained_model='bert-base-uncased'):
        super(BERTSentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits for each class
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(output)
        
        return logits

class BERTMultiClassClassifier(nn.Module):
    """
    BERT model for multi-class sentiment classification (for Amazon reviews)
    """
    def __init__(self, n_classes=3, dropout=0.3, pretrained_model='bert-base-uncased'):
        super(BERTMultiClassClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        
        # Additional layer for better feature extraction
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass for multi-class classification"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.pre_classifier(output)
        output = self.relu(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        
        return logits

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb