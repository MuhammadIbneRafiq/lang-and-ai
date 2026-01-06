"""
E1.1: Majority Class Baseline

Simple baseline that predicts the most frequent class.
Establishes the lower bound for model performance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import Counter

from config import DATA_DIR, RESULTS_DIR, DATASETS, RANDOM_SEED
from utils import compute_metrics, save_results, print_metrics


def run_majority_baseline(dataset_name: str = 'gender'):
    """
    Run majority class baseline for a dataset.
    
    Args:
        dataset_name: 'gender', 'birth_year', or 'political_leaning'
    """
    print(f"\n{'='*60}")
    print(f"E1.1: MAJORITY BASELINE - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    base_path = DATA_DIR / f'E1_baselines/{dataset_name}'
    train_df = pd.read_csv(base_path / 'train.csv')
    dev_df = pd.read_csv(base_path / 'dev.csv')
    test_df = pd.read_csv(base_path / 'test.csv')
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Find majority class
    label_counts = Counter(train_df['label'])
    majority_class = label_counts.most_common(1)[0][0]
    print(f"\nMajority class: '{majority_class}' ({label_counts[majority_class]}/{len(train_df)} = {label_counts[majority_class]/len(train_df)*100:.1f}%)")
    
    # Get all unique labels
    all_labels = sorted(train_df['label'].unique())
    print(f"Label distribution in train:")
    for label in all_labels:
        count = label_counts[label]
        print(f"  {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    # Predict majority class for all samples
    dev_preds = [majority_class] * len(dev_df)
    test_preds = [majority_class] * len(test_df)
    
    # Evaluate
    dev_metrics = compute_metrics(
        dev_df['label'].tolist(), 
        dev_preds,
        label_names=all_labels
    )
    
    test_metrics = compute_metrics(
        test_df['label'].tolist(),
        test_preds,
        label_names=all_labels
    )
    
    print_metrics(dev_metrics, title="Dev Set Metrics")
    print_metrics(test_metrics, title="Test Set Metrics")
    
    # Save results
    results = {
        'model': 'majority_baseline',
        'dataset': dataset_name,
        'majority_class': majority_class,
        'dev_metrics': dev_metrics,
        'test_metrics': test_metrics,
        'label_distribution': dict(label_counts),
    }
    
    output_dir = RESULTS_DIR / f'E1_baselines/{dataset_name}'
    save_results(results, output_dir, 'e1_1_majority')
    
    return results


def main():
    """Run majority baseline on all datasets."""
    all_results = {}
    
    for dataset in DATASETS:
        try:
            results = run_majority_baseline(dataset)
            all_results[dataset] = results
        except FileNotFoundError as e:
            print(f"\nSkipping {dataset}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("MAJORITY BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Dev Accuracy':<15} {'Dev Macro-F1':<15}")
    print("-" * 50)
    for dataset, results in all_results.items():
        dev_acc = results['dev_metrics']['accuracy']
        dev_f1 = results['dev_metrics']['macro_f1']
        print(f"{dataset:<20} {dev_acc:<15.4f} {dev_f1:<15.4f}")


if __name__ == "__main__":
    main()
