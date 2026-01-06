"""
Evaluation metrics for experiments.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import json
from pathlib import Path
from datetime import datetime


def _make_jsonable(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (np.integer,)):
                new_k = int(k)
            elif isinstance(k, (np.floating,)):
                new_k = float(k)
            elif isinstance(k, (np.bool_,)):
                new_k = bool(k)
            else:
                new_k = k
            if not isinstance(new_k, (str, int, float, bool)) and new_k is not None:
                new_k = str(new_k)
            new_dict[new_k] = _make_jsonable(v)
        return new_dict

    if isinstance(obj, (list, tuple)):
        return [_make_jsonable(x) for x in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def compute_metrics(y_true, y_pred, label_names=None):
    """
    Compute comprehensive classification metrics.
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    if label_names:
        report = classification_report(
            y_true, y_pred, 
            target_names=label_names, 
            output_dict=True,
            zero_division=0
        )
        metrics['per_class'] = {
            name: {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support']
            }
            for name in label_names if name in report
        }
    
    return metrics


def compute_performance_drop(metrics_polluted: dict, metrics_cleaned: dict) -> dict:
    """
    Compute absolute and relative performance drops after cleaning.
    
    This is the core metric for H1/H2 in the experiment plan.
    """
    f1_polluted = metrics_polluted['macro_f1']
    f1_cleaned = metrics_cleaned['macro_f1']
    
    absolute_drop = f1_polluted - f1_cleaned
    relative_drop = (absolute_drop / f1_polluted * 100) if f1_polluted > 0 else 0
    
    return {
        'f1_polluted': f1_polluted,
        'f1_cleaned': f1_cleaned,
        'absolute_drop': absolute_drop,
        'relative_drop_percent': relative_drop,
        'accuracy_polluted': metrics_polluted['accuracy'],
        'accuracy_cleaned': metrics_cleaned['accuracy'],
    }


def save_results(results: dict, output_path: Path, experiment_name: str):
    """Save experiment results to JSON."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    results['experiment_name'] = experiment_name
    results['timestamp'] = datetime.now().isoformat()
    
    file_path = output_path / f'{experiment_name}_results.json'
    with open(file_path, 'w') as f:
        json.dump(_make_jsonable(results), f, indent=2, default=str)
    
    print(f"Results saved to: {file_path}")
    return file_path


def print_metrics(metrics: dict, title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Recall:      {metrics['macro_recall']:.4f}")
    
    if 'per_class' in metrics:
        print(f"\n  Per-class F1:")
        for label, class_metrics in metrics['per_class'].items():
            print(f"    {label}: {class_metrics['f1']:.4f} (n={class_metrics['support']})")


def get_confusion_matrix(y_true, y_pred, label_names=None):
    """Get confusion matrix as a dictionary for JSON serialization."""
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        'matrix': cm.tolist(),
        'labels': label_names if label_names else list(range(len(cm)))
    }
    
    return result
