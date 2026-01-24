"""
BERT Sentiment Analysis Training Script with MLflow Tracking
(Memory-safe version for 8GB RAM)
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
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import load_data, create_data_loader
from src.models.bert_classifier import BERTSentimentClassifier, count_parameters, get_model_size_mb


class SentimentTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])

        # Load data
        self.load_datasets()

        # Model
        self.model = BERTSentimentClassifier(
            n_classes=config["num_classes"],
            dropout=config["dropout"],
            pretrained_model=config["model_name"]
        ).to(self.device)

        # 🔒 FREEZE BERT ENCODER (CRITICAL)
        for param in self.model.bert.parameters():
            param.requires_grad = False

        # Model stats
        total_params = count_parameters(self.model)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"📊 Total parameters: {total_params:,}")
        print(f"🔥 Trainable parameters: {trainable_params:,}")
        print(f"💾 Model size: {get_model_size_mb(self.model):.2f} MB")

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # ✅ Optimizer ONLY for trainable params
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        total_steps = len(self.train_loader) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps
        )

        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0

    def load_datasets(self):
        print("\n📂 Loading datasets...")

        train_texts, train_labels = load_data(self.config["train_data"])
        val_texts, val_labels = load_data(self.config["val_data"])
        test_texts, test_labels = load_data(self.config["test_data"])

        print(f"  Train: {len(train_texts)}")
        print(f"  Val:   {len(val_texts)}")
        print(f"  Test:  {len(test_texts)}")

        self.train_loader = create_data_loader(
            train_texts, train_labels, self.tokenizer,
            self.config["batch_size"], self.config["max_length"], shuffle=True
        )
        self.val_loader = create_data_loader(
            val_texts, val_labels, self.tokenizer,
            self.config["batch_size"], self.config["max_length"], shuffle=False
        )
        self.test_loader = create_data_loader(
            test_texts, test_labels, self.tokenizer,
            self.config["batch_size"], self.config["max_length"], shuffle=False
        )

    def train_epoch(self):
        self.model.train()
        losses, correct, total = [], 0, 0

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return np.mean(losses), correct / total

    def evaluate(self, loader):
        self.model.eval()
        losses, preds_all, labels_all = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average="weighted")
        return np.mean(losses), acc, f1, preds_all, labels_all

    def train(self):
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs\n")

        mlflow.set_experiment(self.config["experiment_name"])

        with mlflow.start_run(run_name=f"bert_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(self.config)

            for epoch in range(self.config["epochs"]):
                print(f"\n{'='*50}\nEpoch {epoch+1}/{self.config['epochs']}\n{'='*50}")

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1
                }, step=epoch)

                print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_val_f1 = val_f1
                    self.save_model("best_model.pth")
                    print("✅ Best model saved")

            print("\n🏁 Final evaluation on test set")
            self.load_model("best_model.pth")
            test_loss, test_acc, test_f1, preds, labels = self.evaluate(self.test_loader)

            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(classification_report(labels, preds))

            mlflow.log_metrics({
                "test_accuracy": test_acc,
                "test_f1": test_f1
            })

            mlflow.pytorch.log_model(self.model, "model")

    def save_model(self, name):
        os.makedirs("models/checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/checkpoints/{name}")

    def load_model(self, name):
        self.model.load_state_dict(
            torch.load(f"models/checkpoints/{name}", map_location=self.device)
        )


# ---------------- CONFIG ----------------
config = {
    "train_data": "data/processed/train.csv",
    "val_data": "data/processed/val.csv",
    "test_data": "data/processed/test.csv",

    "model_name": "bert-base-uncased",
    "num_classes": 2,
    "dropout": 0.3,

    # 🔥 MEMORY SAFE
    "max_length": 128,
    "batch_size": 4,

    "epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,

    "experiment_name": "bert_sentiment_analysis"
}


if __name__ == "__main__":
    print("=" * 50)
    print("BERT SENTIMENT ANALYSIS TRAINING")
    print("=" * 50)

    trainer = SentimentTrainer(config)
    trainer.train()
