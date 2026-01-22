"""
Custom PyTorch Dataset for BERT Sentiment Analysis
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis with BERT tokenization
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of sentiment labels (0 or 1)
            tokenizer: Pre-trained BERT tokenizer
            max_length: Maximum sequence length for BERT
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    texts = df['text'].values
    labels = df['sentiment'].values
    return texts, labels

def create_data_loader(texts, labels, tokenizer, batch_size, max_length=512, shuffle=True):
    """
    Create PyTorch DataLoader
    
    Args:
        texts: List of texts
        labels: List of labels
        tokenizer: BERT tokenizer
        batch_size: Batch size for training
        max_length: Max sequence length
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader object
    """
    dataset = SentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )