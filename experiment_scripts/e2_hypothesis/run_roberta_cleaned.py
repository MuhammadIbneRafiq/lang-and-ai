"""
E2.2: RoBERTa on Cleaned Data

Fine-tune RoBERTa on cleaned data to test H1.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import torch
import json

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, BEST_TRANSFORMER_CONFIG, RANDOM_SEED
from utils import compute_metrics, compute_performance_drop, save_results, print_metrics
from e1_baselines.run_roberta import train_roberta, evaluate, set_seed, TextDataset
from torch.utils.data import DataLoader


def run_roberta_cleaned(dataset_name: str = 'gender', save_model: bool = True):
    """Train RoBERTa on cleaned data and compare with polluted."""
    set_seed(RANDOM_SEED)
    
    print(f"\n{'='*60}")
    print(f"E2.2: RoBERTa ON CLEANED - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load cleaned data
    clean_path = DATA_DIR / f'E2_hypothesis/{dataset_name}/cleaned'
    train_df = pd.read_csv(clean_path / 'train.csv')
    dev_df = pd.read_csv(clean_path / 'dev.csv')
    test_df = pd.read_csv(clean_path / 'test.csv')
    
    print(f"\nCleaned data: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")
    
    # Create label map
    all_labels = sorted([str(l) for l in train_df['label'].unique()])
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    
    # Train
    print("\nTraining RoBERTa on cleaned data...")
    config = BEST_TRANSFORMER_CONFIG
    model, tokenizer, history = train_roberta(train_df, dev_df, label_map, **config)
    
    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_labels = [label_map[str(l)] for l in dev_df['label']]
    dev_dataset = TextDataset(dev_df['text'].tolist(), dev_labels, tokenizer, config['max_length'])
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'])
    
    dev_preds, dev_true = evaluate(model, dev_loader, device)
    idx_to_label = {v: k for k, v in label_map.items()}
    dev_pred_labels = [idx_to_label[p] for p in dev_preds]
    dev_true_labels = [idx_to_label[l] for l in dev_true]
    
    dev_metrics = compute_metrics(dev_true_labels, dev_pred_labels, label_names=all_labels)
    print_metrics(dev_metrics, title="Dev Set (Cleaned)")
    
    # Compare with polluted
    polluted_path = RESULTS_DIR / f'E1_baselines/{dataset_name}/e1_4_roberta_polluted_results.json'
    
    if polluted_path.exists():
        with open(polluted_path) as f:
            polluted_results = json.load(f)
        
        drop = compute_performance_drop(polluted_results['dev_metrics'], dev_metrics)
        
        print(f"\n{'='*50}")
        print("PERFORMANCE DROP ANALYSIS (H1)")
        print(f"{'='*50}")
        print(f"  Polluted F1: {drop['f1_polluted']:.4f}")
        print(f"  Cleaned F1:  {drop['f1_cleaned']:.4f}")
        print(f"  Absolute drop: {drop['absolute_drop']:.4f}")
        print(f"  Relative drop: {drop['relative_drop_percent']:.1f}%")
        
        if drop['relative_drop_percent'] >= 10:
            print("  ✓ H1 SUPPORTED: Drop >= 10 percentage points")
        else:
            print("  ✗ H1 NOT SUPPORTED: Drop < 10 percentage points")
    else:
        drop = None
        print("\n⚠️ Run E1.4 first!")
    
    # Save model
    if save_model:
        model_path = MODELS_DIR / f'E2_hypothesis/{dataset_name}/roberta_cleaned'
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        with open(model_path / 'label_map.json', 'w') as f:
            json.dump(label_map, f)
        print(f"\nModel saved to: {model_path}")
    
    results = {
        'model': 'roberta-base',
        'dataset': dataset_name,
        'condition': 'cleaned',
        'dev_metrics': dev_metrics,
        'performance_drop': drop,
    }
    
    output_dir = RESULTS_DIR / f'E2_hypothesis/{dataset_name}'
    save_results(results, output_dir, 'e2_2_roberta_cleaned')
    
    return model, tokenizer, results


def main():
    for dataset in DATASETS:
        try:
            run_roberta_cleaned(dataset)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Error with {dataset}: {e}")


if __name__ == "__main__":
    main()
