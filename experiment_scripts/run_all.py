"""
Run All Experiments

Master script to run all experiments in order.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, MODELS_DIR, RESULTS_DIR


def run_all(dataset: str = 'gender'):
    """Run all experiments for a dataset."""
    
    print("\n" + "="*70)
    print(f"RUNNING ALL EXPERIMENTS FOR: {dataset.upper()}")
    print("="*70)
    
    # Phase 1: Baselines
    print("\n" + "-"*50)
    print("PHASE 1: BASELINES (E1)")
    print("-"*50)
    
    from e1_baselines.run_majority import run_majority_baseline
    from e1_baselines.run_keyword import run_keyword_baseline
    from e1_baselines.run_svm import run_svm_baseline
    from e1_baselines.run_roberta import run_roberta_baseline
    
    print("\n[E1.1] Majority baseline...")
    run_majority_baseline(dataset)
    
    if dataset == 'gender':
        print("\n[E1.2] Keyword baseline...")
        run_keyword_baseline(dataset)
    
    print("\n[E1.3] SVM baseline (polluted)...")
    run_svm_baseline(dataset, condition='polluted', save_model=True)
    
    print("\n[E1.4] RoBERTa baseline (polluted)...")
    run_roberta_baseline(dataset, condition='polluted', save_model=True)
    
    # Phase 2: Hypothesis Testing
    print("\n" + "-"*50)
    print("PHASE 2: HYPOTHESIS TESTING (E2)")
    print("-"*50)
    
    from e2_hypothesis.run_svm_cleaned import run_svm_cleaned
    from e2_hypothesis.run_roberta_cleaned import run_roberta_cleaned
    from e2_hypothesis.run_cross_condition import cross_condition_svm, cross_condition_roberta
    
    print("\n[E2.1] SVM on cleaned...")
    run_svm_cleaned(dataset, save_model=True)
    
    print("\n[E2.2] RoBERTa on cleaned...")
    run_roberta_cleaned(dataset, save_model=True)
    
    print("\n[E2.3] Cross-condition SVM...")
    cross_condition_svm(dataset)
    
    print("\n[E2.4] Cross-condition RoBERTa...")
    cross_condition_roberta(dataset)
    
    # Phase 4: Final Evaluation
    print("\n" + "-"*50)
    print("PHASE 4: FINAL EVALUATION (E4)")
    print("-"*50)
    
    from e4_final.run_final_eval import final_eval_svm, final_eval_roberta
    
    print("\n[E4.1] Final SVM evaluation...")
    final_eval_svm(dataset)
    
    print("\n[E4.2] Final RoBERTa evaluation...")
    final_eval_roberta(dataset)
    
    print("\n" + "="*70)
    print(f"ALL EXPERIMENTS COMPLETE FOR: {dataset.upper()}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--dataset', type=str, default='gender',
                        choices=DATASETS, help='Dataset to run experiments on')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    
    args = parser.parse_args()
    
    if args.all:
        for dataset in DATASETS:
            run_all(dataset)
    else:
        run_all(args.dataset)


if __name__ == "__main__":
    main()
