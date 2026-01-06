"""
Fast Dataset Preparation Script for Label Leakage Experiments
Uses vectorized operations for speed.
"""

import pandas as pd
import numpy as np
import re
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

DATA_DIR = Path("data")
OUTPUT_DIR = Path("experiments")
RANDOM_SEED = 42

# Smaller samples for faster processing
DATASETS_CONFIG = {
    'gender': {
        'file': 'gender.csv',
        'balance_strategy': 'undersample',
        'target_per_class': 8000,
        'max_load': 50000,
    },
    'birth_year': {
        'file': 'birth_year.csv',
        'balance_strategy': 'decade_grouping',
        'target_per_class': 2500,
        'max_load': 50000,
    },
    'political_leaning': {
        'file': 'political_leaning.csv',
        'balance_strategy': 'undersample',
        'target_per_class': 6000,
        'max_load': 50000,
    },
}

# Combined regex pattern for speed
LEAK_PATTERN = re.compile(
    r'\b\d{1,2}\s*[MFmf]\b|'  # 18F, 22M
    r'\b[MFmf]\s*\d{1,2}\b|'  # F18, M22
    r"(?:[Ii]'?m|[Ii]\s+am|im)\s+\d{1,2}\b|"  # I'm 18
    r"(?:[Ii]'?m|[Ii]\s+am|im)\s+(?:a\s+)?(?:male|female|man|woman|guy|girl|dude|lady)\b|"  # I'm a male
    r'\b\d{1,2}\s*(?:years?\s*old|yo|y\.o\.)\b|'  # 18 years old
    r'\bborn\s+in\s+\d{4}\b|'  # born in 1995
    r'\bas\s+a\s+(?:male|female|man|woman|guy|girl)\b|'  # as a woman
    r'[\[\(]\d{1,2}\s*[MFmf][\]\)]|'  # [18F], (22M)
    r'[\[\(][MFmf]\s*\d{1,2}[\]\)]',  # [F18], (M22)
    re.IGNORECASE
)


def create_folders():
    """Create experiment folder structure."""
    folders = [
        'P0_data_prep/gender/polluted', 'P0_data_prep/gender/cleaned',
        'P0_data_prep/birth_year/polluted', 'P0_data_prep/birth_year/cleaned',
        'P0_data_prep/political_leaning/polluted', 'P0_data_prep/political_leaning/cleaned',
        'E1_baselines/gender', 'E1_baselines/birth_year', 'E1_baselines/political_leaning',
        'E2_hypothesis/gender/polluted', 'E2_hypothesis/gender/cleaned',
        'E2_hypothesis/birth_year/polluted', 'E2_hypothesis/birth_year/cleaned',
        'E2_hypothesis/political_leaning/polluted', 'E2_hypothesis/political_leaning/cleaned',
        'E3_ablations/gender', 'E3_ablations/birth_year',
        'E4_final/gender', 'E4_final/birth_year', 'E4_final/political_leaning',
    ]
    for f in folders:
        (OUTPUT_DIR / f).mkdir(parents=True, exist_ok=True)
    print("Created folder structure.")


def decade_label(year):
    """Convert birth year to decade."""
    try:
        y = int(year)
        if y < 1960: return 'pre1960'
        if y >= 2000: return '2000s'
        return f'{(y//10)*10}s'
    except: return None


def _load_text_label_csv(file_path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load CSVs that are either:
    - 3-column with header: author_ID, post, <label>
    - 2-column without header: text,label
    Returns a df with columns: text,label (both as strings).
    """
    # First try: headered format (most of your files look like this)
    try:
        header_df = pd.read_csv(file_path, nrows=0, encoding_errors='replace')
        cols = [str(c).strip() for c in header_df.columns.tolist()]
        if len(cols) >= 2 and ('post' in cols):
            text_col = 'post'
            label_col = cols[-1]
            df = pd.read_csv(
                file_path,
                usecols=[text_col, label_col],
                nrows=max_rows,
                encoding_errors='replace',
            )
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df['text'] = df['text'].astype(str)
            df['label'] = df['label'].astype(str)
            return df
    except Exception:
        pass

    # Fallback: 2-column, no header
    df = pd.read_csv(
        file_path,
        header=None,
        names=['text', 'label'],
        nrows=max_rows,
        on_bad_lines='skip',
        encoding_errors='replace',
    )
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    return df


def _safe_train_dev_test_split(df: pd.DataFrame):
    """Stratified split if possible; otherwise drop tiny classes and/or fall back."""
    label_counts = df['label'].value_counts()
    too_small = label_counts[label_counts < 2]
    if len(too_small) > 0:
        df = df[~df['label'].isin(too_small.index)].copy()

    if len(df) < 10 or df['label'].nunique() < 2:
        idx = df.index.tolist()
        train_idx, temp_idx = train_test_split(idx, train_size=0.7, random_state=RANDOM_SEED)
        dev_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=RANDOM_SEED)
        return df, train_idx, dev_idx, test_idx

    # If any class is too small for the second stratified split, fall back to non-stratified there.
    train_idx, temp_idx = train_test_split(
        df.index.tolist(),
        train_size=0.7,
        stratify=df['label'],
        random_state=RANDOM_SEED,
    )

    temp_labels = df.loc[temp_idx, 'label']
    if temp_labels.value_counts().min() < 2:
        dev_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=RANDOM_SEED)
    else:
        dev_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            stratify=temp_labels,
            random_state=RANDOM_SEED,
        )

    return df, train_idx, dev_idx, test_idx


def process_dataset(name, config):
    print(f"\n{'='*50}")
    print(f"PROCESSING: {name.upper()}")
    print(f"{'='*50}")
    
    # Load (correctly handles quoted "post" column)
    print(f"Loading {config['file']}...", flush=True)
    file_path = DATA_DIR / config['file']
    df = _load_text_label_csv(file_path, max_rows=config['max_load'])
    print(f"  Loaded {len(df):,} rows", flush=True)
    
    # Basic cleaning - vectorized
    print("Cleaning text...", flush=True)
    df['text'] = df['text'].astype(str).str.replace(r'http\S+', '[URL]', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= 10].copy()
    print(f"  After filtering: {len(df):,} rows", flush=True)
    
    # Decade grouping for birth_year
    if config['balance_strategy'] == 'decade_grouping':
        df['label_original'] = df['label']
        df['label'] = df['label'].apply(decade_label)
        df = df[df['label'].notna()].copy()
        print(f"  After decade grouping: {len(df):,} rows", flush=True)
    
    # Balance by undersampling
    print("Balancing...", flush=True)
    target = config['target_per_class']
    balanced = []
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=RANDOM_SEED)
        balanced.append(subset)
    df = pd.concat(balanced, ignore_index=True)
    # Ensure labels are consistently strings to avoid mixed int/str type errors in stratified split
    df['label'] = df['label'].astype(str)
    print(f"  Balanced: {len(df):,} rows", flush=True)
    print(f"  Distribution: {df['label'].value_counts().to_dict()}", flush=True)
    
    # Audit leakage
    print("Auditing leakage...", flush=True)
    sample = df.head(2000)
    leak_rate_before = sample['text'].astype(str).str.contains(LEAK_PATTERN, regex=True).mean() * 100
    print(f"  Leakage rate (before): {leak_rate_before:.1f}%", flush=True)
    
    # Create polluted and cleaned versions
    df_polluted = df.copy()
    print("Depolluting...", flush=True)
    df['text_clean'] = (
        df['text']
        .astype(str)
        .str.replace(LEAK_PATTERN, '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df_cleaned = df[['text_clean', 'label']].copy()
    df_cleaned.columns = ['text', 'label']
    
    # Re-filter cleaned
    df_cleaned['word_count'] = df_cleaned['text'].str.split().str.len()
    df_cleaned = df_cleaned[df_cleaned['word_count'] >= 5].drop(columns=['word_count'])
    
    # Audit after
    leak_rate_after = df_cleaned.head(2000)['text'].astype(str).str.contains(LEAK_PATTERN, regex=True).mean() * 100
    print(f"  Leakage rate (after): {leak_rate_after:.1f}%", flush=True)
    
    # Align indices
    common = df_polluted.index.intersection(df_cleaned.index)
    df_polluted = df_polluted.loc[common, ['text', 'label']]
    df_cleaned = df_cleaned.loc[common]
    print(f"  Aligned samples: {len(df_polluted):,}", flush=True)
    
    # Split (robust to tiny classes)
    print("Creating splits...", flush=True)
    df_polluted, train_idx, dev_idx, test_idx = _safe_train_dev_test_split(df_polluted)
    df_cleaned = df_cleaned.loc[df_polluted.index]
    
    print(f"  Train: {len(train_idx)}, Dev: {len(dev_idx)}, Test: {len(test_idx)}", flush=True)
    
    # Save files
    print("Saving files...", flush=True)
    
    # P0
    df_polluted.to_csv(OUTPUT_DIR / f'P0_data_prep/{name}/polluted/full.csv', index=False)
    df_cleaned.to_csv(OUTPUT_DIR / f'P0_data_prep/{name}/cleaned/full.csv', index=False)
    with open(OUTPUT_DIR / f'P0_data_prep/{name}/leakage_audit.json', 'w') as f:
        json.dump({'leak_rate_before': leak_rate_before, 'leak_rate_after': leak_rate_after}, f)
    
    # E1 - baselines (polluted)
    df_polluted.loc[train_idx].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/train.csv', index=False)
    df_polluted.loc[dev_idx].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/dev.csv', index=False)
    df_polluted.loc[test_idx].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/test.csv', index=False)
    
    # E2 - hypothesis (both)
    df_polluted.loc[train_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/train.csv', index=False)
    df_polluted.loc[dev_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/dev.csv', index=False)
    df_polluted.loc[test_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/test.csv', index=False)
    df_cleaned.loc[train_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/train.csv', index=False)
    df_cleaned.loc[dev_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/dev.csv', index=False)
    df_cleaned.loc[test_idx].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/test.csv', index=False)
    
    # E4 - final test
    df_polluted.loc[test_idx].to_csv(OUTPUT_DIR / f'E4_final/{name}/test_polluted.csv', index=False)
    df_cleaned.loc[test_idx].to_csv(OUTPUT_DIR / f'E4_final/{name}/test_cleaned.csv', index=False)
    
    print(f"  Done saving {name}!", flush=True)
    
    return {
        'dataset': name,
        'total': len(df_polluted),
        'train': len(train_idx),
        'dev': len(dev_idx),
        'test': len(test_idx),
        'leak_before': f"{leak_rate_before:.1f}%",
        'leak_after': f"{leak_rate_after:.1f}%",
    }


def main():
    print("="*50)
    print("FAST DATASET PREPARATION")
    print("="*50, flush=True)
    
    create_folders()
    
    results = []
    for name, config in DATASETS_CONFIG.items():
        results.append(process_dataset(name, config))
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / 'preparation_summary.csv', index=False)
    
    print("\nAll datasets prepared!")
    print(f"  Folder: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
