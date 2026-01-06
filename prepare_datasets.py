"""
Dataset Preparation Script for Label Leakage Experiments
Creates cleaned, balanced, and split datasets organized by experiment phase.

Folder Structure:
experiments/
├── P0_data_prep/           # Phase 0: Raw data audit and validation
│   ├── gender/
│   │   ├── polluted/       # Original data with leaking tokens
│   │   └── cleaned/        # Depolluted data
│   └── birth_year/
├── E1_baselines/           # Phase 1: Baseline experiments
│   ├── gender/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── ...
├── E2_hypothesis/          # Phase 2: Core hypothesis testing
├── E3_ablations/           # Phase 3: Robustness ablations
└── E4_final/               # Phase 4: Final evaluation
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("experiments")
RANDOM_SEED = 42

# Datasets to process (primary focus on gender, secondary on birth_year)
DATASETS_CONFIG = {
    'gender': {
        'file': 'gender.csv',
        'task': 'binary',
        'balance_strategy': 'undersample',
        'target_samples_per_class': 10000,  # Reduced for faster processing
    },
    'birth_year': {
        'file': 'birth_year.csv',
        'task': 'multiclass',
        'balance_strategy': 'decade_grouping',
        'target_samples_per_class': 3000,
    },
    'political_leaning': {
        'file': 'political_leaning.csv',
        'task': 'multiclass',
        'balance_strategy': 'undersample',
        'target_samples_per_class': 8000,
    },
}

# Demographic token patterns for depollution
DEMOGRAPHIC_PATTERNS = {
    # Age-gender compact tokens: "18F", "22M", "F18", "M25"
    'age_gender_compact': r'\b(\d{1,2})\s*[MFmf]\b|\b[MFmf]\s*(\d{1,2})\b',
    
    # "I'm 18", "I am 25", "im 30"
    'i_am_age': r"(?:[Ii]'?m|[Ii]\s+am|im)\s+(\d{1,2})\b",
    
    # "I'm a male/female/man/woman/guy/girl"
    'i_am_gender': r"(?:[Ii]'?m|[Ii]\s+am|im)\s+(?:a\s+)?(male|female|man|woman|guy|girl|dude|lady)\b",
    
    # "18 years old", "25yo", "30 y.o."
    'age_years_old': r'\b(\d{1,2})\s*(?:years?\s*old|yo|y\.o\.)\b',
    
    # "born in 1995"
    'born_in': r'\bborn\s+in\s+(\d{4})\b',
    
    # "as a woman/man/female/male"
    'as_a_gender': r'\bas\s+a\s+(male|female|man|woman|guy|girl)\b',
    
    # Explicit age mentions in context
    'my_age_is': r'\bmy\s+age\s+is\s+(\d{1,2})\b',
    
    # "I'm male/female" without article
    'i_am_gender_direct': r"(?:[Ii]'?m|[Ii]\s+am)\s+(male|female)\b",
    
    # Bracket patterns: [18F], (22M)
    'bracket_age_gender': r'[\[\(](\d{1,2})\s*[MFmf][\]\)]|[\[\(][MFmf]\s*(\d{1,2})[\]\)]',
    
    # "X year old male/female"
    'age_year_old_gender': r'\b(\d{1,2})\s*(?:year\s*old|yo|y\.o\.)\s*(male|female|man|woman|guy|girl)\b',
}


def create_folder_structure():
    """Create the experiment folder structure."""
    folders = [
        'P0_data_prep/gender/polluted',
        'P0_data_prep/gender/cleaned',
        'P0_data_prep/birth_year/polluted',
        'P0_data_prep/birth_year/cleaned',
        'P0_data_prep/political_leaning/polluted',
        'P0_data_prep/political_leaning/cleaned',
        'E1_baselines/gender',
        'E1_baselines/birth_year',
        'E1_baselines/political_leaning',
        'E2_hypothesis/gender/polluted',
        'E2_hypothesis/gender/cleaned',
        'E2_hypothesis/birth_year/polluted',
        'E2_hypothesis/birth_year/cleaned',
        'E3_ablations/gender',
        'E3_ablations/birth_year',
        'E4_final/gender',
        'E4_final/birth_year',
    ]
    
    for folder in folders:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Created folder structure in {OUTPUT_DIR}/")
    return folders


def detect_leaking_tokens(text):
    """Detect all label-leaking tokens in text. Returns dict of pattern: matches."""
    found = {}
    text_str = str(text)
    for name, pattern in DEMOGRAPHIC_PATTERNS.items():
        matches = re.findall(pattern, text_str, re.IGNORECASE)
        if matches:
            found[name] = matches
    return found


def remove_leaking_tokens(text, replacement=''):
    """Remove all demographic label-leaking tokens from text."""
    text_str = str(text)
    for pattern in DEMOGRAPHIC_PATTERNS.values():
        text_str = re.sub(pattern, replacement, text_str, flags=re.IGNORECASE)
    # Clean up extra whitespace
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str


def clean_text_basic(text):
    """Basic text cleaning (URLs, excessive whitespace, etc.)."""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very long repeated characters
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    return text.strip()


def group_birth_year_to_decade(year):
    """Group birth year into decade labels."""
    try:
        year = int(year)
        if year < 1960:
            return 'pre_1960'
        elif year >= 2005:
            return '2000s'
        else:
            decade = (year // 10) * 10
            return f'{decade}s'
    except (ValueError, TypeError):
        return 'unknown'


def load_and_sample_dataset(dataset_name, config, max_samples=None):
    """Load dataset with optional sampling for memory efficiency."""
    import sys
    filepath = DATA_DIR / config['file']
    print(f"\nLoading {dataset_name} from {filepath}...", flush=True)
    
    # For large files, read in chunks with early stopping
    chunks = []
    rows_loaded = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(filepath, header=None, names=['text', 'label'], 
                             chunksize=20000, on_bad_lines='skip'):
        chunk_num += 1
        rows_loaded += len(chunk)
        chunks.append(chunk)
        print(f"  Chunk {chunk_num}: {rows_loaded:,} rows loaded...", flush=True)
        
        if max_samples and rows_loaded >= max_samples:
            print(f"  Reached target samples, stopping early.", flush=True)
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_SEED)
    
    print(f"  Final loaded: {len(df):,} samples", flush=True)
    print(f"  Unique labels: {df['label'].nunique()}", flush=True)
    
    return df


def balance_dataset(df, config, dataset_name):
    """Balance dataset according to configuration."""
    strategy = config['balance_strategy']
    target_per_class = config['target_samples_per_class']
    
    print(f"\nBalancing {dataset_name} with strategy: {strategy}")
    print(f"  Original distribution:")
    print(f"  {df['label'].value_counts().head(10).to_dict()}")
    
    if strategy == 'undersample':
        # Undersample majority classes to target
        balanced_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) > target_per_class:
                label_df = label_df.sample(n=target_per_class, random_state=RANDOM_SEED)
            balanced_dfs.append(label_df)
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        # Ensure labels are consistently strings to avoid mixed int/str type errors in stratified split
        df_balanced['label'] = df_balanced['label'].astype(str)
        
    elif strategy == 'decade_grouping':
        # Group birth years into decades, then balance
        df['label_original'] = df['label']
        df['label'] = df['label'].apply(group_birth_year_to_decade)
        df = df[df['label'] != 'unknown']
        
        # Now undersample each decade
        balanced_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) > target_per_class:
                label_df = label_df.sample(n=target_per_class, random_state=RANDOM_SEED)
            balanced_dfs.append(label_df)
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        # Ensure labels are consistently strings to avoid mixed int/str type errors in stratified split
        df_balanced['label'] = df_balanced['label'].astype(str)
        
    else:
        df_balanced = df
    
    print(f"  Balanced distribution:")
    print(f"  {df_balanced['label'].value_counts().to_dict()}")
    print(f"  Total samples: {len(df_balanced):,}")
    
    return df_balanced


def create_splits(df, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """Create stratified train/dev/test splits."""
    # First split: train vs (dev+test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],
        random_state=RANDOM_SEED
    )
    
    # Second split: dev vs test
    relative_dev_ratio = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=relative_dev_ratio,
        stratify=temp_df['label'],
        random_state=RANDOM_SEED
    )
    
    print(f"  Splits: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
    
    return train_df, dev_df, test_df


def audit_leakage(df, dataset_name):
    """Audit dataset for label leaking prevalence."""
    print(f"\nAuditing leakage in {dataset_name}...")
    
    leaking_counts = Counter()
    samples_with_leakage = 0
    total_checked = min(5000, len(df))  # Check sample for speed
    
    sample_df = df.head(total_checked)
    
    for _, row in sample_df.iterrows():
        leaks = detect_leaking_tokens(row['text'])
        if leaks:
            samples_with_leakage += 1
            for pattern_name in leaks.keys():
                leaking_counts[pattern_name] += 1
    
    leakage_rate = (samples_with_leakage / total_checked) * 100
    
    audit_results = {
        'dataset': dataset_name,
        'total_checked': total_checked,
        'samples_with_leakage': samples_with_leakage,
        'leakage_rate_percent': round(leakage_rate, 2),
        'pattern_counts': dict(leaking_counts),
    }
    
    print(f"  Leakage rate: {leakage_rate:.1f}%")
    print(f"  Pattern breakdown: {dict(leaking_counts.most_common(5))}")
    
    return audit_results


def process_dataset(dataset_name, config):
    """Full processing pipeline for a dataset."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data (limit for memory)
    max_samples = config['target_samples_per_class'] * 10  # Load more than needed
    df = load_and_sample_dataset(dataset_name, config, max_samples=max_samples)
    
    # Basic text cleaning
    print("\nApplying basic text cleaning...")
    df['text'] = df['text'].apply(clean_text_basic)
    
    # Filter out very short texts
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df = df[df['word_count'] >= 10]
    print(f"  After filtering short texts: {len(df):,} samples")
    
    # Audit leakage before cleaning
    audit_before = audit_leakage(df, f"{dataset_name}_polluted")
    
    # Balance dataset
    df_balanced = balance_dataset(df, config, dataset_name)
    
    # Create polluted version (balanced but not depolluted)
    df_polluted = df_balanced.copy()
    
    # Create cleaned version (depolluted)
    print(f"\nDepolluting {dataset_name}...")
    df_cleaned = df_balanced.copy()
    df_cleaned['text'] = df_cleaned['text'].apply(remove_leaking_tokens)
    
    # Re-filter after cleaning (some texts may become too short)
    df_cleaned['word_count'] = df_cleaned['text'].apply(lambda x: len(str(x).split()))
    df_cleaned = df_cleaned[df_cleaned['word_count'] >= 10]
    print(f"  After depollution: {len(df_cleaned):,} samples")
    
    # Audit leakage after cleaning
    audit_after = audit_leakage(df_cleaned, f"{dataset_name}_cleaned")
    
    # Align polluted and cleaned to have same samples
    common_indices = df_polluted.index.intersection(df_cleaned.index)
    df_polluted = df_polluted.loc[common_indices]
    df_cleaned = df_cleaned.loc[common_indices]
    print(f"  Aligned samples: {len(df_polluted):,}")
    
    # Create splits
    print(f"\nCreating train/dev/test splits...")
    
    # Use same indices for both polluted and cleaned
    train_idx, temp_idx = train_test_split(
        df_polluted.index.tolist(),
        train_size=0.7,
        stratify=df_polluted['label'],
        random_state=RANDOM_SEED
    )
    dev_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        stratify=df_polluted.loc[temp_idx, 'label'],
        random_state=RANDOM_SEED
    )
    
    splits = {
        'train': train_idx,
        'dev': dev_idx,
        'test': test_idx,
    }
    
    # Save to experiment folders
    save_experiment_data(dataset_name, df_polluted, df_cleaned, splits, audit_before, audit_after)
    
    return {
        'dataset': dataset_name,
        'total_samples': len(df_polluted),
        'train_samples': len(train_idx),
        'dev_samples': len(dev_idx),
        'test_samples': len(test_idx),
        'audit_before': audit_before,
        'audit_after': audit_after,
    }


def save_experiment_data(dataset_name, df_polluted, df_cleaned, splits, audit_before, audit_after):
    """Save processed data to experiment folders."""
    
    # Columns to save
    save_cols = ['text', 'label']
    if 'label_original' in df_polluted.columns:
        save_cols.append('label_original')
    
    # P0: Data prep - full polluted and cleaned
    print(f"\nSaving to P0_data_prep/{dataset_name}/...")
    df_polluted[save_cols].to_csv(
        OUTPUT_DIR / f'P0_data_prep/{dataset_name}/polluted/full.csv', 
        index=False
    )
    df_cleaned[save_cols].to_csv(
        OUTPUT_DIR / f'P0_data_prep/{dataset_name}/cleaned/full.csv',
        index=False
    )
    
    # Save audit results
    with open(OUTPUT_DIR / f'P0_data_prep/{dataset_name}/leakage_audit.json', 'w') as f:
        json.dump({'before': audit_before, 'after': audit_after}, f, indent=2)
    
    # E1: Baselines - polluted splits
    print(f"Saving to E1_baselines/{dataset_name}/...")
    for split_name, indices in splits.items():
        df_polluted.loc[indices, save_cols].to_csv(
            OUTPUT_DIR / f'E1_baselines/{dataset_name}/{split_name}.csv',
            index=False
        )
    
    # E2: Hypothesis testing - both polluted and cleaned splits
    print(f"Saving to E2_hypothesis/{dataset_name}/...")
    for split_name, indices in splits.items():
        df_polluted.loc[indices, save_cols].to_csv(
            OUTPUT_DIR / f'E2_hypothesis/{dataset_name}/polluted/{split_name}.csv',
            index=False
        )
        df_cleaned.loc[indices, save_cols].to_csv(
            OUTPUT_DIR / f'E2_hypothesis/{dataset_name}/cleaned/{split_name}.csv',
            index=False
        )
    
    # E3: Ablations - same as E2 for now (scripts will create partial cleaning)
    # E4: Final - test sets only
    print(f"Saving to E4_final/{dataset_name}/...")
    df_polluted.loc[splits['test'], save_cols].to_csv(
        OUTPUT_DIR / f'E4_final/{dataset_name}/test_polluted.csv',
        index=False
    )
    df_cleaned.loc[splits['test'], save_cols].to_csv(
        OUTPUT_DIR / f'E4_final/{dataset_name}/test_cleaned.csv',
        index=False
    )
    
    print(f"  Saved all splits for {dataset_name}")


def main():
    print("="*60)
    print("DATASET PREPARATION FOR LABEL LEAKAGE EXPERIMENTS")
    print("="*60)
    
    # Create folder structure
    create_folder_structure()
    
    # Process each dataset
    results = []
    for dataset_name, config in DATASETS_CONFIG.items():
        result = process_dataset(dataset_name, config)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PREPARATION SUMMARY")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Dataset': r['dataset'],
            'Total': r['total_samples'],
            'Train': r['train_samples'],
            'Dev': r['dev_samples'],
            'Test': r['test_samples'],
            'Leakage Before': f"{r['audit_before']['leakage_rate_percent']}%",
            'Leakage After': f"{r['audit_after']['leakage_rate_percent']}%",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(OUTPUT_DIR / 'preparation_summary.csv', index=False)
    
    with open(OUTPUT_DIR / 'preparation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FOLDER STRUCTURE CREATED:")
    print(f"{'='*60}")
    print("""
experiments/
├── P0_data_prep/           # Full datasets + leakage audit
│   ├── gender/polluted/    
│   ├── gender/cleaned/     
│   └── ...
├── E1_baselines/           # Polluted train/dev/test splits
│   ├── gender/train.csv    
│   ├── gender/dev.csv      
│   └── gender/test.csv     
├── E2_hypothesis/          # Both polluted & cleaned splits
│   ├── gender/polluted/    
│   └── gender/cleaned/     
├── E3_ablations/           # For ablation experiments
└── E4_final/               # Final test sets
    """)
    
    print("\nDone! Run your experiments using the prepared datasets.")


if __name__ == "__main__":
    main()
