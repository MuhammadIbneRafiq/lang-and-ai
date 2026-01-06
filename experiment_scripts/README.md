# Experiment Scripts

Scripts for running the label leakage experiments.

## Structure

```
experiment_scripts/
├── config.py              # Central configuration
├── run_all.py             # Master script to run everything
├── utils/
│   ├── data_loader.py     # Dataset loading utilities
│   └── metrics.py         # Evaluation metrics
├── e1_baselines/          # Phase 1: Baseline experiments
│   ├── run_majority.py    # E1.1: Majority class baseline
│   ├── run_keyword.py     # E1.2: Keyword heuristic
│   ├── run_svm.py         # E1.3: Stylometric SVM
│   └── run_roberta.py     # E1.4: RoBERTa baseline
├── e2_hypothesis/         # Phase 2: Core hypothesis testing
│   ├── run_svm_cleaned.py      # E2.1: SVM on cleaned data
│   ├── run_roberta_cleaned.py  # E2.2: RoBERTa on cleaned data
│   └── run_cross_condition.py  # E2.3/E2.4: Cross-condition eval
└── e4_final/              # Phase 4: Final evaluation
    └── run_final_eval.py  # E4.1/E4.2: Test set evaluation
```

## Quick Start

### Run all experiments for gender dataset:
```bash
python run_all.py --dataset gender
```

### Run all experiments for all datasets:
```bash
python run_all.py --all
```

### Run individual experiments:
```bash
# Baselines
python e1_baselines/run_majority.py
python e1_baselines/run_keyword.py
python e1_baselines/run_svm.py
python e1_baselines/run_roberta.py

# Hypothesis testing
python e2_hypothesis/run_svm_cleaned.py
python e2_hypothesis/run_roberta_cleaned.py
python e2_hypothesis/run_cross_condition.py

# Final evaluation
python e4_final/run_final_eval.py
```

## Requirements

```
torch
transformers
scikit-learn
pandas
numpy
tqdm
joblib
```

## Output

- **Models**: Saved to `models/` directory
- **Results**: JSON files in `results/` directory

## Experiment Mapping to exp.md

| Script | Experiment ID | Description |
|--------|--------------|-------------|
| run_majority.py | E1.1 | Majority class baseline |
| run_keyword.py | E1.2 | Keyword heuristic |
| run_svm.py | E1.3 | Stylometric SVM (polluted) |
| run_roberta.py | E1.4 | RoBERTa (polluted) |
| run_svm_cleaned.py | E2.1 | SVM on cleaned |
| run_roberta_cleaned.py | E2.2 | RoBERTa on cleaned |
| run_cross_condition.py | E2.3/E2.4 | Cross-condition eval |
| run_final_eval.py | E4.1/E4.2 | Final test evaluation |
