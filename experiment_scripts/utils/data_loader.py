"""
Data loading utilities for experiments.
"""

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(base_path: Path, split: str = 'train') -> pd.DataFrame:
    """Load a dataset split from CSV."""
    file_path = base_path / f'{split}.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def load_experiment_data(
    experiment_dir: Path,
    dataset_name: str,
    condition: str = 'polluted'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/dev/test splits for an experiment.
    
    Args:
        experiment_dir: Base experiments directory
        dataset_name: 'gender', 'birth_year', or 'political_leaning'
        condition: 'polluted' or 'cleaned'
    
    Returns:
        Tuple of (train_df, dev_df, test_df)
    """
    if condition in ['polluted', 'cleaned']:
        base = experiment_dir / f'E2_hypothesis/{dataset_name}/{condition}'
    else:
        base = experiment_dir / f'E1_baselines/{dataset_name}'
    
    train_df = load_dataset(base, 'train')
    dev_df = load_dataset(base, 'dev')
    test_df = load_dataset(base, 'test')
    
    return train_df, dev_df, test_df


def encode_labels(df: pd.DataFrame, label_map: dict = None) -> tuple[list, dict]:
    """
    Encode string labels to integers.
    
    Returns:
        Tuple of (encoded_labels, label_map)
    """
    if label_map is None:
        unique_labels = sorted(df['label'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    encoded = [label_map[label] for label in df['label']]
    return encoded, label_map


def get_label_map(train_df: pd.DataFrame) -> dict:
    """Get label mapping from training data."""
    unique_labels = sorted(train_df['label'].unique())
    return {label: idx for idx, label in enumerate(unique_labels)}
