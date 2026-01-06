"""
E2.3/E2.4: Cross-Condition Evaluation

Test models trained on polluted data against cleaned test sets.
This evaluates generalization without retraining.
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

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, BEST_TRANSFORMER_CONFIG
from utils import TextDataset, compute_metrics, compute_performance_drop, save_results, print_metrics
from e1_baselines.run_roberta import evaluate


def cross_condition_svm(dataset_name: str = 'gender'):
    """Evaluate polluted-trained SVM on cleaned data."""
    print(f"\n{'='*60}")
    print(f"E2.3: CROSS-CONDITION SVM - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load polluted-trained model
    model_path = MODELS_DIR / f'E1_baselines/{dataset_name}/svm_polluted.joblib'
    if not model_path.exists():
        print(f"⚠️ Model not found: {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Load cleaned test data
    clean_path = DATA_DIR / f'E2_hypothesis/{dataset_name}/cleaned'
    dev_df = pd.read_csv(clean_path / 'dev.csv')
    
    dev_texts = dev_df['text'].tolist()
    dev_labels = [str(l) for l in dev_df['label']]
    all_labels = sorted(set(dev_labels))
    
    # Predict
    dev_preds = model.predict(dev_texts)
    dev_metrics = compute_metrics(dev_labels, dev_preds, label_names=all_labels)
    print_metrics(dev_metrics, title="Polluted→Cleaned (SVM)")
    
    results = {'model': 'svm_polluted_to_cleaned', 'dev_metrics': dev_metrics}
    output_dir = RESULTS_DIR / f'E2_hypothesis/{dataset_name}'
    save_results(results, output_dir, 'e2_3_svm_cross')
    
    return dev_metrics


def cross_condition_roberta(dataset_name: str = 'gender'):
    """Evaluate polluted-trained RoBERTa on cleaned data."""
    print(f"\n{'='*60}")
    print(f"E2.4: CROSS-CONDITION RoBERTa - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load polluted-trained model
    model_path = MODELS_DIR / f'E1_baselines/{dataset_name}/roberta_polluted'
    if not model_path.exists():
        print(f"⚠️ Model not found: {model_path}")
        return None
    
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    with open(model_path / 'label_map.json') as f:
        label_map = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Load cleaned data
    clean_path = DATA_DIR / f'E2_hypothesis/{dataset_name}/cleaned'
    dev_df = pd.read_csv(clean_path / 'dev.csv')
    
    dev_labels = [label_map[str(l)] for l in dev_df['label']]
    dev_dataset = TextDataset(dev_df['text'].tolist(), dev_labels, tokenizer, 128)
    dev_loader = DataLoader(dev_dataset, batch_size=16)
    
    dev_preds, dev_true = evaluate(model, dev_loader, device)
    
    idx_to_label = {v: k for k, v in label_map.items()}
    dev_pred_labels = [idx_to_label[p] for p in dev_preds]
    dev_true_labels = [idx_to_label[l] for l in dev_true]
    
    dev_metrics = compute_metrics(dev_true_labels, dev_pred_labels, label_names=list(label_map.keys()))
    print_metrics(dev_metrics, title="Polluted→Cleaned (RoBERTa)")
    
    results = {'model': 'roberta_polluted_to_cleaned', 'dev_metrics': dev_metrics}
    output_dir = RESULTS_DIR / f'E2_hypothesis/{dataset_name}'
    save_results(results, output_dir, 'e2_4_roberta_cross')
    
    return dev_metrics


def main():
    for dataset in DATASETS:
        try:
            cross_condition_svm(dataset)
            cross_condition_roberta(dataset)
        except Exception as e:
            print(f"Error with {dataset}: {e}")


if __name__ == "__main__":
    main()
