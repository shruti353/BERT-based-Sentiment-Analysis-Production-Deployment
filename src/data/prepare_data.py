"""
Data Preparation Script for BERT Sentiment Analysis
Downloads and preprocesses the IMDB dataset for quick start
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

class SentimentDataPreparator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_imdb_dataset(self):
        """Download IMDB dataset using HuggingFace datasets"""
        print("📥 Downloading IMDB dataset...")
        
        # Load dataset
        dataset = load_dataset("imdb")
        
        # Convert to pandas for easier manipulation
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        print(f"✅ Downloaded {len(train_df)} training samples")
        print(f"✅ Downloaded {len(test_df)} test samples")
        
        return train_df, test_df
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove HTML tags
        text = text.replace("<br />", " ")
        text = text.replace("<br>", " ")
        # Remove extra whitespace
        text = " ".join(text.split())
        return text
    
    def preprocess_dataset(self, df, dataset_type="train"):
        """Preprocess the dataset"""
        print(f"\n🔧 Preprocessing {dataset_type} dataset...")
        
        # Clean text
        tqdm.pandas(desc="Cleaning text")
        df['text'] = df['text'].progress_apply(self.clean_text)
        
        # Rename label column for clarity
        df = df.rename(columns={'label': 'sentiment'})
        
        # Add text length feature
        df['text_length'] = df['text'].apply(len)
        
        # Filter out extremely short or long reviews (optional)
        original_len = len(df)
        df = df[(df['text_length'] >= 50) & (df['text_length'] <= 5000)]
        print(f"  Filtered {original_len - len(df)} samples (too short/long)")
        
        return df
    
    def create_splits(self, train_df, test_df, val_size=0.15):
        """Create train/val/test splits"""
        print(f"\n📊 Creating data splits...")
        
        # Split training data into train and validation
        train_data, val_data = train_test_split(
            train_df, 
            test_size=val_size, 
            random_state=42,
            stratify=train_df['sentiment']
        )
        
        # Use provided test set
        test_data = test_df
        
        print(f"  Train: {len(train_data)} samples")
        print(f"  Val:   {len(val_data)} samples")
        print(f"  Test:  {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def save_datasets(self, train_df, val_df, test_df):
        """Save processed datasets"""
        print("\n💾 Saving processed datasets...")
        
        train_df.to_csv(
            os.path.join(self.processed_dir, "train.csv"), 
            index=False
        )
        val_df.to_csv(
            os.path.join(self.processed_dir, "val.csv"), 
            index=False
        )
        test_df.to_csv(
            os.path.join(self.processed_dir, "test.csv"), 
            index=False
        )
        
        # Save dataset statistics
        stats = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'num_classes': train_df['sentiment'].nunique(),
            'class_distribution_train': train_df['sentiment'].value_counts().to_dict(),
            'avg_text_length': {
                'train': float(train_df['text_length'].mean()),
                'val': float(val_df['text_length'].mean()),
                'test': float(test_df['text_length'].mean())
            }
        }
        
        with open(os.path.join(self.processed_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved to {self.processed_dir}/")
        return stats
    
    def print_statistics(self, stats):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("📈 DATASET STATISTICS")
        print("="*50)
        print(f"Train samples: {stats['train_size']:,}")
        print(f"Val samples:   {stats['val_size']:,}")
        print(f"Test samples:  {stats['test_size']:,}")
        print(f"\nNumber of classes: {stats['num_classes']}")
        print(f"\nClass distribution (train):")
        for label, count in stats['class_distribution_train'].items():
            label_name = "Positive" if label == 1 else "Negative"
            percentage = (count / stats['train_size']) * 100
            print(f"  {label_name} ({label}): {count:,} ({percentage:.1f}%)")
        print(f"\nAverage text length:")
        print(f"  Train: {stats['avg_text_length']['train']:.0f} characters")
        print(f"  Val:   {stats['avg_text_length']['val']:.0f} characters")
        print(f"  Test:  {stats['avg_text_length']['test']:.0f} characters")
        print("="*50)
    
    def run_pipeline(self):
        """Run complete data preparation pipeline"""
        print("🎬 Starting Data Preparation Pipeline\n")
        
        # Download data
        train_df, test_df = self.download_imdb_dataset()
        
        # Preprocess
        train_df = self.preprocess_dataset(train_df, "train")
        test_df = self.preprocess_dataset(test_df, "test")
        
        # Create splits
        train_data, val_data, test_data = self.create_splits(train_df, test_df)
        
        # Save datasets
        stats = self.save_datasets(train_data, val_data, test_data)
        
        # Print statistics
        self.print_statistics(stats)
        
        print("\n✅ Data preparation complete!")
        print(f"📁 Processed data saved to: {self.processed_dir}/")
        
        return stats

if __name__ == "__main__":
    preparator = SentimentDataPreparator()
    preparator.run_pipeline()