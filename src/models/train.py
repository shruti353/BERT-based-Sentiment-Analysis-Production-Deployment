"""
BERT Sentiment Analysis Training Script with MLflow Tracking
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import load_data, create_data_loader
from src.models.bert_classifier import BERTSentimentClassifier, count_parameters, get_model_size_mb

class SentimentTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        
        # Load data
        self.load_datasets()
        
        # Initialize model
        self.model = BERTSentimentClassifier(
            n_classes=config['num_classes'],
            dropout=config['dropout'],
            pretrained_model=config['model_name']
        ).to(self.device)
        
        print(f"📊 Model parameters: {count_parameters(self.model):,}")
        print(f"💾 Model size: {get_model_size_mb(self.model):.2f} MB")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Tracking
        self.best_val_accuracy = 0
        self.best_val_f1 = 0
        self.train_losses = []
        self.val_losses = []
    
    def load_datasets(self):
        """Load train, validation, and test datasets"""
        print("\n📂 Loading datasets...")
        
        # Load data
        train_texts, train_labels = load_data(self.config['train_data'])
        val_texts, val_labels = load_data(self.config['val_data'])
        test_texts, test_labels = load_data(self.config['test_data'])
        
        print(f"  Train: {len(train_texts)} samples")
        print(f"  Val:   {len(val_texts)} samples")
        print(f"  Test:  {len(test_texts)} samples")
        
        # Create data loaders
        self.train_loader = create_data_loader(
            train_texts, train_labels, self.tokenizer,
            self.config['batch_size'], self.config['max_length'], shuffle=True
        )
        self.val_loader = create_data_loader(
            val_texts, val_labels, self.tokenizer,
            self.config['batch_size'], self.config['max_length'], shuffle=False
        )
        self.test_loader = create_data_loader(
            test_texts, test_labels, self.tokenizer,
            self.config['batch_size'], self.config['max_length'], shuffle=False
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        losses = []
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': np.mean(losses),
                'acc': (correct_predictions.double() / total_predictions).item()
            })
        
        return np.mean(losses), (correct_predictions.double() / total_predictions).item()
    
    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                losses.append(loss.item())
                _, preds = torch.max(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return np.mean(losses), accuracy, f1, all_preds, all_labels
    
    def train(self):
        """Complete training loop"""
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...\n")
        
        # Start MLflow run
        mlflow.set_experiment(self.config['experiment_name'])
        
        with mlflow.start_run(run_name=f"bert_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(self.config)
            mlflow.log_param("model_parameters", count_parameters(self.model))
            mlflow.log_param("model_size_mb", f"{get_model_size_mb(self.model):.2f}")
            mlflow.log_param("device", str(self.device))
            
            for epoch in range(self.config['epochs']):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.config['epochs']}")
                print(f"{'='*60}")
                
                # Train
                train_loss, train_acc = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, val_acc, val_f1, val_preds, val_labels = self.evaluate(self.val_loader)
                self.val_losses.append(val_loss)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_f1_score': val_f1,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }, step=epoch)
                
                # Print epoch summary
                print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
                
                # Save best model
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_val_f1 = val_f1
                    self.save_model('best_model.pth')
                    print(f"✅ New best model saved! (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
            
            # Final evaluation on test set
            print(f"\n{'='*60}")
            print("Final Evaluation on Test Set")
            print(f"{'='*60}")
            
            # Load best model
            self.load_model('best_model.pth')
            test_loss, test_acc, test_f1, test_preds, test_labels = self.evaluate(self.test_loader)
            
            # Generate classification report
            report = classification_report(test_labels, test_preds, target_names=['Negative', 'Positive'])
            cm = confusion_matrix(test_labels, test_preds)
            
            print(f"\nTest Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"\nClassification Report:\n{report}")
            print(f"\nConfusion Matrix:\n{cm}")
            
            # Log final metrics
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_f1_score': test_f1,
                'best_val_accuracy': self.best_val_accuracy,
                'best_val_f1': self.best_val_f1
            })
            
            # Save artifacts
            mlflow.log_artifact('models/checkpoints/best_model.pth')
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            print(f"\n✅ Training complete!")
            print(f"🏆 Best Val Accuracy: {self.best_val_accuracy:.4f}")
            print(f"🏆 Test Accuracy: {test_acc:.4f}")
            print(f"🏆 Test F1 Score: {test_f1:.4f}")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join('models/checkpoints', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }, checkpoint_path)
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join('models/checkpoints', filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

# Training configuration
config = {
    # Data
    'train_data': 'data/processed/train.csv',
    'val_data': 'data/processed/val.csv',
    'test_data': 'data/processed/test.csv',
    
    # Model
    'model_name': 'bert-base-uncased',
    'num_classes': 2,
    'dropout': 0.3,
    'max_length': 512,
    
    # Training
    'epochs': 4,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    
    # MLflow
    'experiment_name': 'bert_sentiment_analysis'
}

if __name__ == "__main__":
    print("="*60)
    print("BERT SENTIMENT ANALYSIS TRAINING")
    print("="*60)
    
    # Create trainer and start training
    trainer = SentimentTrainer(config)
    trainer.train()