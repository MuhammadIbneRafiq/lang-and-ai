"""
E1.2: Keyword Heuristic Baseline

Rule-based baseline using regex patterns to predict demographics.
Tests whether simple pattern matching can solve the task.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
from collections import Counter

from config import DATA_DIR, RESULTS_DIR, DATASETS, RANDOM_SEED
from utils import compute_metrics, save_results, print_metrics


# Gender prediction patterns
GENDER_PATTERNS = {
    'female': [
        r'\b\d{1,2}\s*[Ff]\b',           # 18F
        r'\b[Ff]\s*\d{1,2}\b',           # F18
        r"(?:i'?m|i\s+am)\s+(?:a\s+)?(?:female|woman|girl|lady)\b",
        r'\bas\s+a\s+(?:female|woman|girl|mother|mom)\b',
        r'\bmy\s+(?:husband|boyfriend)\b',  # Implies female speaker
        r'\b(?:pregnant|my\s+period)\b',
        r'\[\d{1,2}\s*[Ff]\]',
        r'\(\d{1,2}\s*[Ff]\)',
    ],
    'male': [
        r'\b\d{1,2}\s*[Mm]\b',           # 22M
        r'\b[Mm]\s*\d{1,2}\b',           # M22
        r"(?:i'?m|i\s+am)\s+(?:a\s+)?(?:male|man|guy|dude)\b",
        r'\bas\s+a\s+(?:male|man|guy|father|dad)\b',
        r'\bmy\s+(?:wife|girlfriend)\b',  # Implies male speaker
        r'\[\d{1,2}\s*[Mm]\]',
        r'\(\d{1,2}\s*[Mm]\)',
    ],
}

# Compile patterns
COMPILED_GENDER_PATTERNS = {
    gender: [re.compile(p, re.IGNORECASE) for p in patterns]
    for gender, patterns in GENDER_PATTERNS.items()
}


def predict_gender_keyword(text: str) -> tuple[str, float]:
    """
    Predict gender using keyword patterns.
    
    Returns:
        Tuple of (prediction, confidence)
        If no match, returns (None, 0.0)
    """
    text = str(text).lower()
    
    scores = {'female': 0, 'male': 0}
    
    for gender, patterns in COMPILED_GENDER_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            scores[gender] += len(matches)
    
    if scores['female'] > scores['male']:
        return '0', scores['female']  # 0 = female in dataset
    elif scores['male'] > scores['female']:
        return '1', scores['male']    # 1 = male in dataset
    else:
        return None, 0.0


def run_keyword_baseline(dataset_name: str = 'gender'):
    """
    Run keyword heuristic baseline.
    Currently only supports gender dataset.
    """
    print(f"\n{'='*60}")
    print(f"E1.2: KEYWORD HEURISTIC - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if dataset_name != 'gender':
        print(f"  ⚠️ Keyword baseline only implemented for 'gender' dataset")
        print(f"  Skipping {dataset_name}...")
        return None
    
    # Load data
    base_path = DATA_DIR / f'E1_baselines/{dataset_name}'
    train_df = pd.read_csv(base_path / 'train.csv')
    dev_df = pd.read_csv(base_path / 'dev.csv')
    test_df = pd.read_csv(base_path / 'test.csv')
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Get majority class for fallback
    majority_class = str(Counter(train_df['label']).most_common(1)[0][0])
    
    # Predict on dev set
    dev_preds = []
    dev_matched = 0
    
    for text in dev_df['text']:
        pred, conf = predict_gender_keyword(text)
        if pred is not None:
            dev_preds.append(pred)
            dev_matched += 1
        else:
            dev_preds.append(majority_class)  # Fallback to majority
    
    # Predict on test set
    test_preds = []
    test_matched = 0
    
    for text in test_df['text']:
        pred, conf = predict_gender_keyword(text)
        if pred is not None:
            test_preds.append(pred)
            test_matched += 1
        else:
            test_preds.append(majority_class)
    
    print(f"\nKeyword matching coverage:")
    print(f"  Dev:  {dev_matched}/{len(dev_df)} ({dev_matched/len(dev_df)*100:.1f}%)")
    print(f"  Test: {test_matched}/{len(test_df)} ({test_matched/len(test_df)*100:.1f}%)")
    
    # Convert labels to strings for comparison
    dev_labels = [str(l) for l in dev_df['label']]
    test_labels = [str(l) for l in test_df['label']]
    all_labels = sorted(set(dev_labels))
    
    # Evaluate
    dev_metrics = compute_metrics(dev_labels, dev_preds, label_names=all_labels)
    test_metrics = compute_metrics(test_labels, test_preds, label_names=all_labels)
    
    print_metrics(dev_metrics, title="Dev Set Metrics")
    print_metrics(test_metrics, title="Test Set Metrics")
    
    # Analyze matched vs unmatched accuracy
    matched_correct = sum(1 for i, (pred, label) in enumerate(zip(dev_preds, dev_labels)) 
                          if predict_gender_keyword(dev_df.iloc[i]['text'])[0] is not None and pred == label)
    matched_total = dev_matched
    
    if matched_total > 0:
        print(f"\nMatched samples accuracy: {matched_correct}/{matched_total} = {matched_correct/matched_total*100:.1f}%")
    
    # Save results
    results = {
        'model': 'keyword_heuristic',
        'dataset': dataset_name,
        'dev_coverage': dev_matched / len(dev_df),
        'test_coverage': test_matched / len(test_df),
        'dev_metrics': dev_metrics,
        'test_metrics': test_metrics,
        'fallback_class': majority_class,
    }
    
    output_dir = RESULTS_DIR / f'E1_baselines/{dataset_name}'
    save_results(results, output_dir, 'e1_2_keyword')
    
    return results


def main():
    """Run keyword baseline on gender dataset."""
    run_keyword_baseline('gender')


if __name__ == "__main__":
    main()
