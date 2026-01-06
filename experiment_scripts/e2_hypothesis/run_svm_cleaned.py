"""
E2: Core Hypothesis Testing - SVM on Cleaned Data

Train and evaluate stylometric SVM on cleaned (depolluted) data.
Compare with polluted baseline (E1.3) to test H2.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import joblib

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, SVM_CONFIG, RANDOM_SEED
from utils import compute_metrics, compute_performance_drop, save_results, print_metrics
from e1_baselines.run_svm import train_svm


def run_svm_cleaned(dataset_name: str = 'gender', save_model: bool = True):
    """Train SVM on cleaned data and compare with polluted."""
    print(f"\n{'='*60}")
    print(f"E2.1: SVM ON CLEANED - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load cleaned data
    clean_path = DATA_DIR / f'E2_hypothesis/{dataset_name}/cleaned'
    train_df = pd.read_csv(clean_path / 'train.csv')
    dev_df = pd.read_csv(clean_path / 'dev.csv')
    test_df = pd.read_csv(clean_path / 'test.csv')
    
    print(f"\nCleaned data from: {clean_path}")
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    # Prepare data
    train_texts = train_df['text'].tolist()
    train_labels = [str(l) for l in train_df['label']]
    dev_texts = dev_df['text'].tolist()
    dev_labels = [str(l) for l in dev_df['label']]
    all_labels = sorted(set(train_labels))
    
    # Train
    print("\nTraining SVM on cleaned data...")
    model = train_svm(train_texts, train_labels, **SVM_CONFIG)
    
    # Evaluate
    dev_preds = model.predict(dev_texts)
    dev_metrics = compute_metrics(dev_labels, dev_preds, label_names=all_labels)
    print_metrics(dev_metrics, title="Dev Set (Cleaned)")
    
    # Load polluted results for comparison
    polluted_results_path = RESULTS_DIR / f'E1_baselines/{dataset_name}/e1_3_svm_polluted_results.json'
    
    if polluted_results_path.exists():
        import json
        with open(polluted_results_path) as f:
            polluted_results = json.load(f)
        
        drop = compute_performance_drop(polluted_results['dev_metrics'], dev_metrics)
        
        print(f"\n{'='*50}")
        print("PERFORMANCE DROP ANALYSIS (H2)")
        print(f"{'='*50}")
        print(f"  Polluted F1: {drop['f1_polluted']:.4f}")
        print(f"  Cleaned F1:  {drop['f1_cleaned']:.4f}")
        print(f"  Absolute drop: {drop['absolute_drop']:.4f}")
        print(f"  Relative drop: {drop['relative_drop_percent']:.1f}%")
    else:
        drop = None
        print("\n⚠️ Run E1.3 (polluted SVM) first for comparison!")
    
    # Save model
    if save_model:
        model_path = MODELS_DIR / f'E2_hypothesis/{dataset_name}'
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path / 'svm_cleaned.joblib')
        print(f"\nModel saved to: {model_path / 'svm_cleaned.joblib'}")
    
    # Save results
    results = {
        'model': 'stylometric_svm',
        'dataset': dataset_name,
        'condition': 'cleaned',
        'dev_metrics': dev_metrics,
        'performance_drop': drop,
    }
    
    output_dir = RESULTS_DIR / f'E2_hypothesis/{dataset_name}'
    save_results(results, output_dir, 'e2_1_svm_cleaned')
    
    return model, results


def main():
    for dataset in DATASETS:
        try:
            run_svm_cleaned(dataset)
        except FileNotFoundError as e:
            print(f"Skipping {dataset}: {e}")


if __name__ == "__main__":
    main()
