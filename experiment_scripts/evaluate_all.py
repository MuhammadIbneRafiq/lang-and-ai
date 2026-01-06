"""
Evaluation Summary Script

Aggregates all experiment results and produces comparison tables.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import pandas as pd
from config import RESULTS_DIR, DATASETS


def load_all_results():
    """Load all saved experiment results."""
    results = {}
    
    for json_file in RESULTS_DIR.rglob('*.json'):
        rel_path = json_file.relative_to(RESULTS_DIR)
        key = str(rel_path).replace('\\', '/').replace('_results.json', '')
        
        with open(json_file) as f:
            results[key] = json.load(f)
    
    return results


def create_summary_table(dataset: str = 'gender'):
    """Create comparison table for a dataset."""
    results = load_all_results()
    
    rows = []
    
    # E1 Baselines
    experiments = [
        ('E1.1 Majority', f'E1_baselines/{dataset}/e1_1_majority'),
        ('E1.2 Keyword', f'E1_baselines/{dataset}/e1_2_keyword'),
        ('E1.3 SVM (poll)', f'E1_baselines/{dataset}/e1_3_svm_polluted'),
        ('E1.4 RoBERTa (poll)', f'E1_baselines/{dataset}/e1_4_roberta_polluted'),
        ('E2.1 SVM (clean)', f'E2_hypothesis/{dataset}/e2_1_svm_cleaned'),
        ('E2.2 RoBERTa (clean)', f'E2_hypothesis/{dataset}/e2_2_roberta_cleaned'),
        ('E2.3 SVM cross', f'E2_hypothesis/{dataset}/e2_3_svm_cross'),
        ('E2.4 RoBERTa cross', f'E2_hypothesis/{dataset}/e2_4_roberta_cross'),
    ]
    
    for name, key in experiments:
        if key in results:
            r = results[key]
            dev_metrics = r.get('dev_metrics', {})
            test_metrics = r.get('test_metrics', {})
            
            row = {
                'Experiment': name,
                'Dev Acc': dev_metrics.get('accuracy', '-'),
                'Dev F1': dev_metrics.get('macro_f1', '-'),
                'Test Acc': test_metrics.get('accuracy', '-') if test_metrics else '-',
                'Test F1': test_metrics.get('macro_f1', '-') if test_metrics else '-',
            }
            
            # Add performance drop if available
            drop = r.get('performance_drop')
            if drop:
                row['F1 Drop'] = f"{drop['absolute_drop']:.3f}"
                row['Drop %'] = f"{drop['relative_drop_percent']:.1f}%"
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def create_hypothesis_summary(dataset: str = 'gender'):
    """Create H1/H2 hypothesis test summary."""
    results = load_all_results()
    
    print(f"\n{'='*70}")
    print(f"HYPOTHESIS TESTING SUMMARY - {dataset.upper()}")
    print(f"{'='*70}")
    
    # H1: RoBERTa drop >= 10%
    roberta_clean_key = f'E2_hypothesis/{dataset}/e2_2_roberta_cleaned'
    if roberta_clean_key in results:
        drop = results[roberta_clean_key].get('performance_drop', {})
        if drop:
            rel_drop = drop.get('relative_drop_percent', 0)
            print(f"\nH1: RoBERTa performance drop after cleaning")
            print(f"    Polluted F1: {drop['f1_polluted']:.4f}")
            print(f"    Cleaned F1:  {drop['f1_cleaned']:.4f}")
            print(f"    Relative drop: {rel_drop:.1f}%")
            print(f"    Hypothesis (drop >= 10%): {'✓ SUPPORTED' if rel_drop >= 10 else '✗ NOT SUPPORTED'}")
    
    # H2: SVM drop < 50% of RoBERTa drop
    svm_clean_key = f'E2_hypothesis/{dataset}/e2_1_svm_cleaned'
    if svm_clean_key in results and roberta_clean_key in results:
        svm_drop = results[svm_clean_key].get('performance_drop', {})
        roberta_drop = results[roberta_clean_key].get('performance_drop', {})
        
        if svm_drop and roberta_drop:
            svm_rel = svm_drop.get('relative_drop_percent', 0)
            roberta_rel = roberta_drop.get('relative_drop_percent', 0)
            
            print(f"\nH2: SVM vs RoBERTa robustness comparison")
            print(f"    SVM relative drop:     {svm_rel:.1f}%")
            print(f"    RoBERTa relative drop: {roberta_rel:.1f}%")
            
            if roberta_rel > 0:
                ratio = svm_rel / roberta_rel * 100
                print(f"    SVM drop as % of RoBERTa: {ratio:.1f}%")
                print(f"    Hypothesis (SVM < 50% of RoBERTa): {'✓ SUPPORTED' if ratio < 50 else '✗ NOT SUPPORTED'}")


def print_full_summary():
    """Print complete evaluation summary."""
    
    print("\n" + "="*70)
    print("EXPERIMENT EVALUATION SUMMARY")
    print("="*70)
    
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        table = create_summary_table(dataset)
        
        if len(table) > 0:
            # Format numeric columns
            for col in ['Dev Acc', 'Dev F1', 'Test Acc', 'Test F1']:
                if col in table.columns:
                    table[col] = table[col].apply(
                        lambda x: f"{x:.4f}" if isinstance(x, float) else x
                    )
            
            print(table.to_string(index=False))
            
            create_hypothesis_summary(dataset)
        else:
            print("  No results found. Run experiments first!")
    
    # Save to CSV
    output_file = RESULTS_DIR / 'evaluation_summary.csv'
    all_tables = []
    for dataset in DATASETS:
        table = create_summary_table(dataset)
        if len(table) > 0:
            table.insert(0, 'Dataset', dataset)
            all_tables.append(table)
    
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"\n\nSummary saved to: {output_file}")


def main():
    print_full_summary()


if __name__ == "__main__":
    main()
