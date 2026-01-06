"""
Experiment configuration settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "experiments"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASETS = ['gender', 'birth_year', 'political_leaning']
PRIMARY_DATASET = 'gender'  # Main focus per exp.md

# Random seed for reproducibility
RANDOM_SEED = 42

# Training hyperparameters (from exp.md Exp-2-hptune grid)
TRANSFORMER_CONFIG = {
    'learning_rates': [1e-5, 2e-5, 3e-5],
    'batch_sizes': [16, 32],
    'epochs': 3,
    'max_length': 128,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'dropout': 0.1,
}

# Best config (update after hyperparameter tuning)
BEST_TRANSFORMER_CONFIG = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'epochs': 3,
    'max_length': 128,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
}

# SVM configuration (stylometric baseline)
SVM_CONFIG = {
    'ngram_range': (3, 5),  # Character n-grams
    'max_features': 50000,
    'analyzer': 'char_wb',  # Word-boundary aware char n-grams
    'C': 1.0,
}

# Model names
MODELS = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
}

# Experiment phases
EXPERIMENTS = {
    'E1': {
        'name': 'Baselines',
        'scripts': ['e1_majority', 'e1_keyword', 'e1_svm', 'e1_roberta'],
    },
    'E2': {
        'name': 'Hypothesis Testing',
        'scripts': ['e2_svm_cleaned', 'e2_roberta_cleaned', 'e2_cross_condition'],
    },
    'E3': {
        'name': 'Ablations',
        'scripts': ['e3_ablations'],
    },
    'E4': {
        'name': 'Final Evaluation',
        'scripts': ['e4_final_eval'],
    },
}
