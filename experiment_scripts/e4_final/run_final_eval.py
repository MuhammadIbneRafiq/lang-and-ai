"""
E4: Final Evaluation on Held-Out Test Set

Run best models on test set for final results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import torch
import json
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS
from utils import TextDataset, compute_metrics, compute_performance_drop, save_results, print_metrics, get_confusion_matrix
from e1_baselines.run_roberta import evaluate


def final_eval_svm(dataset_name: str = 'gender'):
    """Final SVM evaluation on test set."""
    print(f"\n{'='*60}")
    print(f"E4: FINAL SVM - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    for condition in ['polluted', 'cleaned']:
        if condition == 'polluted':
            model_path = MODELS_DIR / f'E1_baselines/{dataset_name}/svm_polluted.joblib'
        else:
            model_path = MODELS_DIR / f'E2_hypothesis/{dataset_name}/svm_cleaned.joblib'
        
        if not model_path.exists():
            print(f"⚠️ Skipping {condition}: model not found")
            continue
        
        model = joblib.load(model_path)
        
        # Load test set
        test_path = DATA_DIR / f'E4_final/{dataset_name}/test_{condition}.csv'
        test_df = pd.read_csv(test_path)
        
        test_texts = test_df['text'].tolist()
        test_labels = [str(l) for l in test_df['label']]
        all_labels = sorted(set(test_labels))
        
        test_preds = model.predict(test_texts)
        metrics = compute_metrics(test_labels, test_preds, label_names=all_labels)
        
        print_metrics(metrics, title=f"Test Set ({condition})")
        results[condition] = {'metrics': metrics, 'confusion': get_confusion_matrix(test_labels, test_preds, all_labels)}
    
    # Performance drop
    if 'polluted' in results and 'cleaned' in results:
        drop = compute_performance_drop(results['polluted']['metrics'], results['cleaned']['metrics'])
        print(f"\n  Performance drop: {drop['relative_drop_percent']:.1f}%")
        results['performance_drop'] = drop
    
    output_dir = RESULTS_DIR / f'E4_final/{dataset_name}'
    save_results(results, output_dir, 'e4_svm_final')
    return results


def final_eval_roberta(dataset_name: str = 'gender'):
    """Final RoBERTa evaluation on test set."""
    print(f"\n{'='*60}")
    print(f"E4: FINAL RoBERTa - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    for condition in ['polluted', 'cleaned']:
        if condition == 'polluted':
            model_path = MODELS_DIR / f'E1_baselines/{dataset_name}/roberta_polluted'
        else:
            model_path = MODELS_DIR / f'E2_hypothesis/{dataset_name}/roberta_cleaned'
        
        if not model_path.exists():
            print(f"⚠️ Skipping {condition}: model not found")
            continue
        
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model.to(device)
        
        with open(model_path / 'label_map.json') as f:
            label_map = json.load(f)
        
        # Load test set
        test_path = DATA_DIR / f'E4_final/{dataset_name}/test_{condition}.csv'
        test_df = pd.read_csv(test_path)
        
        test_labels = [label_map[str(l)] for l in test_df['label']]
        test_dataset = TextDataset(test_df['text'].tolist(), test_labels, tokenizer, 128)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        test_preds, test_true = evaluate(model, test_loader, device)
        
        idx_to_label = {v: k for k, v in label_map.items()}
        test_pred_labels = [idx_to_label[p] for p in test_preds]
        test_true_labels = [idx_to_label[l] for l in test_true]
        
        metrics = compute_metrics(test_true_labels, test_pred_labels, label_names=list(label_map.keys()))
        print_metrics(metrics, title=f"Test Set ({condition})")
        
        results[condition] = {'metrics': metrics, 'confusion': get_confusion_matrix(test_true_labels, test_pred_labels, list(label_map.keys()))}
    
    if 'polluted' in results and 'cleaned' in results:
        drop = compute_performance_drop(results['polluted']['metrics'], results['cleaned']['metrics'])
        print(f"\n  Performance drop: {drop['relative_drop_percent']:.1f}%")
        results['performance_drop'] = drop
    
    output_dir = RESULTS_DIR / f'E4_final/{dataset_name}'
    save_results(results, output_dir, 'e4_roberta_final')
    return results


def main():
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    
    for dataset in DATASETS:
        try:
            final_eval_svm(dataset)
            final_eval_roberta(dataset)
        except Exception as e:
            print(f"Error with {dataset}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
