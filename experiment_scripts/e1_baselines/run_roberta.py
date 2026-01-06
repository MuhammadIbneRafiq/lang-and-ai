"""
E1.4: RoBERTa-base Baseline

Fine-tuned RoBERTa for author profiling.
This is the main transformer baseline for comparison with stylometric approach.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json

from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, 
    BEST_TRANSFORMER_CONFIG, MODELS, RANDOM_SEED
)
from utils import (
    TextDataset, compute_metrics, save_results, 
    print_metrics, get_label_map
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels


def train_roberta(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    label_map: dict,
    model_name: str = 'roberta-base',
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: str = None,
):
    """
    Fine-tune RoBERTa for sequence classification.
    
    Returns:
        Tuple of (model, tokenizer, training_history)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_map),
    )
    model.to(device)
    
    # Encode labels
    train_labels = [label_map[str(l)] for l in train_df['label']]
    dev_labels = [label_map[str(l)] for l in dev_df['label']]
    
    # Create datasets
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_labels,
        tokenizer,
        max_length=max_length
    )
    dev_dataset = TextDataset(
        dev_df['text'].tolist(),
        dev_labels,
        tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    history = {'train_loss': [], 'dev_metrics': []}
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        history['train_loss'].append(train_loss)
        print(f"  Train loss: {train_loss:.4f}")
        
        # Evaluate on dev
        dev_preds, dev_true = evaluate(model, dev_loader, device)
        
        # Convert back to label names
        idx_to_label = {v: k for k, v in label_map.items()}
        dev_pred_labels = [idx_to_label[p] for p in dev_preds]
        dev_true_labels = [idx_to_label[l] for l in dev_true]
        
        dev_metrics = compute_metrics(
            dev_true_labels, 
            dev_pred_labels, 
            label_names=list(label_map.keys())
        )
        history['dev_metrics'].append(dev_metrics)
        
        print(f"  Dev Accuracy: {dev_metrics['accuracy']:.4f}")
        print(f"  Dev Macro-F1: {dev_metrics['macro_f1']:.4f}")
        
        # Save best model
        if dev_metrics['macro_f1'] > best_f1:
            best_f1 = dev_metrics['macro_f1']
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, tokenizer, history


def run_roberta_baseline(
    dataset_name: str = 'gender',
    condition: str = 'polluted',
    save_model: bool = True,
    config: dict = None
):
    """
    Train and evaluate RoBERTa on a dataset.
    
    Args:
        dataset_name: 'gender', 'birth_year', or 'political_leaning'
        condition: 'polluted' or 'cleaned'
        save_model: Whether to save the trained model
        config: Training configuration (uses BEST_TRANSFORMER_CONFIG if None)
    """
    set_seed(RANDOM_SEED)
    
    if config is None:
        config = BEST_TRANSFORMER_CONFIG
    
    print(f"\n{'='*60}")
    print(f"E1.4: RoBERTa-BASE - {dataset_name.upper()} ({condition})")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    # Load data
    if condition == 'polluted':
        base_path = DATA_DIR / f'E1_baselines/{dataset_name}'
    else:
        base_path = DATA_DIR / f'E2_hypothesis/{dataset_name}/{condition}'
    
    train_df = pd.read_csv(base_path / 'train.csv')
    dev_df = pd.read_csv(base_path / 'dev.csv')
    test_df = pd.read_csv(base_path / 'test.csv')
    
    print(f"\nData loaded from: {base_path}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Create label map
    all_labels = sorted([str(l) for l in train_df['label'].unique()])
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    print(f"  Label map: {label_map}")
    
    # Train model
    print("\nTraining RoBERTa...")
    model, tokenizer, history = train_roberta(
        train_df, 
        dev_df,
        label_map,
        model_name=MODELS['roberta'],
        **config
    )
    
    # Final evaluation on dev and test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dev evaluation
    dev_labels = [label_map[str(l)] for l in dev_df['label']]
    dev_dataset = TextDataset(
        dev_df['text'].tolist(),
        dev_labels,
        tokenizer,
        max_length=config['max_length']
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'])
    dev_preds, dev_true = evaluate(model, dev_loader, device)
    
    # Test evaluation
    test_labels = [label_map[str(l)] for l in test_df['label']]
    test_dataset = TextDataset(
        test_df['text'].tolist(),
        test_labels,
        tokenizer,
        max_length=config['max_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    test_preds, test_true = evaluate(model, test_loader, device)
    
    # Convert predictions back to labels
    idx_to_label = {v: k for k, v in label_map.items()}
    dev_pred_labels = [idx_to_label[p] for p in dev_preds]
    dev_true_labels = [idx_to_label[l] for l in dev_true]
    test_pred_labels = [idx_to_label[p] for p in test_preds]
    test_true_labels = [idx_to_label[l] for l in test_true]
    
    dev_metrics = compute_metrics(dev_true_labels, dev_pred_labels, label_names=all_labels)
    test_metrics = compute_metrics(test_true_labels, test_pred_labels, label_names=all_labels)
    
    print_metrics(dev_metrics, title="Dev Set Metrics (Final)")
    print_metrics(test_metrics, title="Test Set Metrics")
    
    # Save model
    if save_model:
        model_path = MODELS_DIR / f'E1_baselines/{dataset_name}/roberta_{condition}'
        model_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Save label map
        with open(model_path / 'label_map.json', 'w') as f:
            json.dump(label_map, f)
        
        print(f"\nModel saved to: {model_path}")
    
    # Save results
    results = {
        'model': 'roberta-base',
        'dataset': dataset_name,
        'condition': condition,
        'config': config,
        'label_map': label_map,
        'training_history': {
            'train_loss': history['train_loss'],
            'dev_f1_per_epoch': [m['macro_f1'] for m in history['dev_metrics']]
        },
        'dev_metrics': dev_metrics,
        'test_metrics': test_metrics,
    }
    
    output_dir = RESULTS_DIR / f'E1_baselines/{dataset_name}'
    save_results(results, output_dir, f'e1_4_roberta_{condition}')
    
    return model, tokenizer, results


def main():
    """Run RoBERTa baseline on all datasets (polluted)."""
    all_results = {}
    
    for dataset in DATASETS:
        try:
            model, tokenizer, results = run_roberta_baseline(
                dataset, 
                condition='polluted',
                save_model=True
            )
            all_results[dataset] = results
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except FileNotFoundError as e:
            print(f"\nSkipping {dataset}: {e}")
        except Exception as e:
            print(f"\nError with {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("RoBERTa BASELINE SUMMARY (POLLUTED)")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Dev Macro-F1':<15} {'Test Macro-F1':<15}")
    print("-" * 50)
    for dataset, results in all_results.items():
        dev_f1 = results['dev_metrics']['macro_f1']
        test_f1 = results['test_metrics']['macro_f1']
        print(f"{dataset:<20} {dev_f1:<15.4f} {test_f1:<15.4f}")


if __name__ == "__main__":
    main()
