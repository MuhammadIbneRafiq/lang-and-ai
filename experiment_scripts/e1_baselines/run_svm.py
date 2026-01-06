"""
E1.3: Stylometric SVM Baseline

Character n-gram SVM for stylometric author profiling.
Uses character-level features which are theoretically more robust to semantic shortcuts.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, SVM_CONFIG, RANDOM_SEED
from utils import compute_metrics, save_results, print_metrics, get_label_map


def train_svm(
    train_texts: list,
    train_labels: list,
    ngram_range: tuple = (3, 5),
    max_features: int = 50000,
    analyzer: str = 'char_wb',
    C: float = 1.0,
) -> Pipeline:
    """
    Train a character n-gram SVM classifier.
    
    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        ngram_range: Range of n-gram sizes (default 3-5 for char)
        max_features: Maximum vocabulary size
        analyzer: 'char', 'char_wb', or 'word'
        C: SVM regularization parameter
    
    Returns:
        Trained sklearn Pipeline
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True,  # Use log(1+tf) for better scaling
        )),
        ('svm', LinearSVC(
            C=C,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=10000,
        ))
    ])
    
    pipeline.fit(train_texts, train_labels)
    return pipeline


def run_svm_baseline(
    dataset_name: str = 'gender',
    condition: str = 'polluted',
    save_model: bool = True
):
    """
    Train and evaluate SVM on a dataset.
    
    Args:
        dataset_name: 'gender', 'birth_year', or 'political_leaning'
        condition: 'polluted' or 'cleaned'
        save_model: Whether to save the trained model
    """
    print(f"\n{'='*60}")
    print(f"E1.3: STYLOMETRIC SVM - {dataset_name.upper()} ({condition})")
    print(f"{'='*60}")
    
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
    
    # Get labels
    train_texts = train_df['text'].tolist()
    train_labels = [str(l) for l in train_df['label']]
    dev_texts = dev_df['text'].tolist()
    dev_labels = [str(l) for l in dev_df['label']]
    test_texts = test_df['text'].tolist()
    test_labels = [str(l) for l in test_df['label']]
    
    all_labels = sorted(set(train_labels))
    print(f"  Classes: {all_labels}")
    
    # Train model
    print(f"\nTraining SVM...")
    print(f"  Config: {SVM_CONFIG}")
    
    model = train_svm(
        train_texts, 
        train_labels,
        ngram_range=SVM_CONFIG['ngram_range'],
        max_features=SVM_CONFIG['max_features'],
        analyzer=SVM_CONFIG['analyzer'],
        C=SVM_CONFIG['C'],
    )
    
    # Evaluate
    print("Evaluating...")
    dev_preds = model.predict(dev_texts)
    test_preds = model.predict(test_texts)
    
    dev_metrics = compute_metrics(dev_labels, dev_preds, label_names=all_labels)
    test_metrics = compute_metrics(test_labels, test_preds, label_names=all_labels)
    
    print_metrics(dev_metrics, title="Dev Set Metrics")
    print_metrics(test_metrics, title="Test Set Metrics")
    
    # Analyze top features
    vectorizer = model.named_steps['tfidf']
    svm = model.named_steps['svm']
    
    feature_names = vectorizer.get_feature_names_out()
    
    if len(all_labels) == 2:
        # Binary classification - single coefficient vector
        coefs = svm.coef_[0]
        top_pos_idx = np.argsort(coefs)[-10:][::-1]
        top_neg_idx = np.argsort(coefs)[:10]
        
        print(f"\nTop features for class '{all_labels[1]}':")
        for idx in top_pos_idx:
            print(f"  '{feature_names[idx]}': {coefs[idx]:.4f}")
        
        print(f"\nTop features for class '{all_labels[0]}':")
        for idx in top_neg_idx:
            print(f"  '{feature_names[idx]}': {coefs[idx]:.4f}")
    
    # Save model
    if save_model:
        model_path = MODELS_DIR / f'E1_baselines/{dataset_name}'
        model_path.mkdir(parents=True, exist_ok=True)
        model_file = model_path / f'svm_{condition}.joblib'
        joblib.dump(model, model_file)
        print(f"\nModel saved to: {model_file}")
    
    # Save results
    results = {
        'model': 'stylometric_svm',
        'dataset': dataset_name,
        'condition': condition,
        'config': SVM_CONFIG,
        'n_features': len(feature_names),
        'dev_metrics': dev_metrics,
        'test_metrics': test_metrics,
    }
    
    output_dir = RESULTS_DIR / f'E1_baselines/{dataset_name}'
    save_results(results, output_dir, f'e1_3_svm_{condition}')
    
    return model, results


def main():
    """Run SVM baseline on all datasets (polluted)."""
    all_results = {}
    
    for dataset in DATASETS:
        try:
            model, results = run_svm_baseline(dataset, condition='polluted')
            all_results[dataset] = results
        except FileNotFoundError as e:
            print(f"\nSkipping {dataset}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SVM BASELINE SUMMARY (POLLUTED)")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Dev Macro-F1':<15} {'Test Macro-F1':<15}")
    print("-" * 50)
    for dataset, results in all_results.items():
        dev_f1 = results['dev_metrics']['macro_f1']
        test_f1 = results['test_metrics']['macro_f1']
        print(f"{dataset:<20} {dev_f1:<15.4f} {test_f1:<15.4f}")


if __name__ == "__main__":
    main()
