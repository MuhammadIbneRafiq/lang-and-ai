"""
Fast Dataset Preparation Script for Label Leakage Experiments
Uses vectorized operations for speed.

REFINED VERSION with fixes:
1. Author-level splitting (prevents identity leakage)
2. Expanded leakage patterns (husband/wife, pregnant, mother/father, etc.)
3. Less aggressive undersampling (uses more data)
4. Balanced decade grouping for birth_year
5. Lower minimum word count threshold
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import json

DATA_DIR = Path("data")
OUTPUT_DIR = Path("experiments")
RANDOM_SEED = 42

# FIX #3: Increased sample sizes to use more data
DATASETS_CONFIG = {
    'gender': {
        'file': 'gender.csv',
        'balance_strategy': 'undersample',
        'target_per_class': 15000,  # Increased from 8000
        'max_load': 100000,          # Increased from 50000
    },
    'birth_year': {
        'file': 'birth_year.csv',
        'balance_strategy': 'decade_grouping',
        'target_per_class': 5000,    # Increased from 2500
        'max_load': 100000,           # Increased from 50000
    },
    'political_leaning': {
        'file': 'political_leaning.csv',
        'balance_strategy': 'undersample',
        'target_per_class': 10000,   # Increased from 6000
        'max_load': 100000,           # Increased from 50000
    },
}

# FIX #2: Expanded regex pattern with more leakage indicators
LEAK_PATTERN = re.compile(
    # Original patterns
    r'\b\d{1,2}\s*[MFmf]\b|'                    # 18F, 22M
    r'\b[MFmf]\s*\d{1,2}\b|'                    # F18, M22
    r"(?:[Ii]'?m|[Ii]\s+am|im)\s+\d{1,2}\b|"   # I'm 18
    r"(?:[Ii]'?m|[Ii]\s+am|im)\s+(?:a\s+)?(?:male|female|man|woman|guy|girl|dude|lady|boy)\b|"  # I'm a male
    r'\b\d{1,2}\s*(?:years?\s*old|yo|y\.o\.)\b|'  # 18 years old
    r'\bborn\s+in\s+\d{4}\b|'                  # born in 1995
    r'\bas\s+a\s+(?:male|female|man|woman|guy|girl|mother|father|mom|dad)\b|'  # as a woman/mother
    r'[\[\(]\d{1,2}\s*[MFmf][\]\)]|'           # [18F], (22M)
    r'[\[\(][MFmf]\s*\d{1,2}[\]\)]|'           # [F18], (M22)
    
    # NEW patterns for better coverage
    r'\bmy\s+(?:husband|wife|boyfriend|girlfriend)\b|'   # my husband/wife (implies gender)
    r'\b(?:i am|i\'m|im)\s+(?:pregnant|expecting)\b|'    # I'm pregnant (female indicator)
    r'\bmy\s+(?:period|menstrual|pregnancy)\b|'          # my period (female indicator)
    r'\bas\s+(?:a\s+)?(?:mother|father|mom|dad|mum)\b|'  # as a mother/father
    r'\b(?:i am|i\'m|im)\s+(?:a\s+)?(?:mother|father|mom|dad|mum|husband|wife)\b|'  # I'm a mother
    r'\b(?:i am|i\'m|im)\s+(?:a\s+)?(?:he|she|him|her)\b|'  # Sometimes used in self-description
    r'\bmy\s+age\s+is\s+\d{1,2}\b|'            # my age is 25
    r'\b\d{1,2}\s*(?:year\s*old|yo|y\.o\.)\s*(?:male|female|man|woman|guy|girl)\b|'  # 25 year old male
    r'\b(?:male|female)\s+here\b|'              # male here, female here
    r'\b(?:guy|girl|man|woman)\s+here\b',       # guy here
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


# FIX #4: Balanced decade grouping with more equal spans
def decade_label(year):
    """Convert birth year to decade with balanced spans."""
    try:
        y = int(year)
        # More balanced grouping: each group spans ~10 years
        if y < 1970:
            return 'pre1970'      # Before 1970 (older users, likely smaller)
        elif y < 1980:
            return '1970s'        # 1970-1979
        elif y < 1990:
            return '1980s'        # 1980-1989
        elif y < 2000:
            return '1990s'        # 1990-1999
        else:
            return '2000s'        # 2000+ (younger users)
    except:
        return None


def _load_text_label_csv(file_path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load CSVs that are either:
    - 3-column with header: author_ID, post, <label>
    - 2-column without header: text,label
    Returns a df with columns: author_id (if available), text, label.
    """
    # First try: headered format with author_ID (most of your files look like this)
    try:
        header_df = pd.read_csv(file_path, nrows=0, encoding_errors='replace')
        cols = [str(c).strip() for c in header_df.columns.tolist()]
        
        if len(cols) >= 2 and ('post' in cols):
            text_col = 'post'
            label_col = cols[-1]
            
            # Check if author_ID exists
            usecols = [text_col, label_col]
            has_author_id = 'author_ID' in cols or 'author_id' in cols
            author_col = 'author_ID' if 'author_ID' in cols else ('author_id' if 'author_id' in cols else None)
            
            if author_col:
                usecols = [author_col, text_col, label_col]
            
            df = pd.read_csv(
                file_path,
                usecols=usecols,
                nrows=max_rows,
                encoding_errors='replace',
            )
            
            # Rename columns
            rename_map = {text_col: 'text', label_col: 'label'}
            if author_col:
                rename_map[author_col] = 'author_id'
            df = df.rename(columns=rename_map)
            
            df['text'] = df['text'].astype(str)
            df['label'] = df['label'].astype(str)
            
            # If no author_id, create synthetic one (each row = unique author)
            if 'author_id' not in df.columns:
                df['author_id'] = range(len(df))
            
            return df
    except Exception as e:
        print(f"  Warning: Error loading with header: {e}")

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
    # Synthetic author_id for fallback
    df['author_id'] = range(len(df))
    return df


# FIX #1: Author-level splitting to prevent identity leakage
def _author_level_train_dev_test_split(df: pd.DataFrame):
    """
    Stratified split BY AUTHOR to prevent identity leakage.
    All posts from one author go into the same split.
    """
    # Get unique authors with their labels (use first label per author for stratification)
    author_labels = df.groupby('author_id')['label'].first().reset_index()
    
    # Count labels to handle rare classes
    label_counts = author_labels['label'].value_counts()
    rare_labels = label_counts[label_counts < 3].index.tolist()
    
    if rare_labels:
        print(f"  Dropping {len(rare_labels)} rare labels with <3 authors")
        author_labels = author_labels[~author_labels['label'].isin(rare_labels)]
        valid_authors = author_labels['author_id'].tolist()
        df = df[df['author_id'].isin(valid_authors)].copy()
    
    if len(author_labels) < 10 or author_labels['label'].nunique() < 2:
        # Fallback to random split if too few authors
        print("  Warning: Too few authors for stratified split, using random split")
        authors = author_labels['author_id'].tolist()
        train_authors, temp_authors = train_test_split(authors, train_size=0.7, random_state=RANDOM_SEED)
        dev_authors, test_authors = train_test_split(temp_authors, train_size=0.5, random_state=RANDOM_SEED)
    else:
        # Stratified split by author
        try:
            train_authors, temp_authors = train_test_split(
                author_labels['author_id'].tolist(),
                train_size=0.7,
                stratify=author_labels['label'],
                random_state=RANDOM_SEED,
            )
            
            temp_labels = author_labels[author_labels['author_id'].isin(temp_authors)]
            
            # Check if we can stratify the second split
            temp_label_counts = temp_labels['label'].value_counts()
            if temp_label_counts.min() >= 2:
                dev_authors, test_authors = train_test_split(
                    temp_authors,
                    train_size=0.5,
                    stratify=temp_labels.set_index('author_id').loc[temp_authors, 'label'],
                    random_state=RANDOM_SEED,
                )
            else:
                dev_authors, test_authors = train_test_split(
                    temp_authors, train_size=0.5, random_state=RANDOM_SEED
                )
        except Exception as e:
            print(f"  Warning: Stratified split failed ({e}), using random split")
            authors = author_labels['author_id'].tolist()
            train_authors, temp_authors = train_test_split(authors, train_size=0.7, random_state=RANDOM_SEED)
            dev_authors, test_authors = train_test_split(temp_authors, train_size=0.5, random_state=RANDOM_SEED)
    
    # Map authors back to row indices
    train_idx = df[df['author_id'].isin(train_authors)].index.tolist()
    dev_idx = df[df['author_id'].isin(dev_authors)].index.tolist()
    test_idx = df[df['author_id'].isin(test_authors)].index.tolist()
    
    return df, train_idx, dev_idx, test_idx


def process_dataset(name, config):
    print(f"\n{'='*50}")
    print(f"PROCESSING: {name.upper()}")
    print(f"{'='*50}")
    
    # Load (correctly handles quoted "post" column and author_ID)
    print(f"Loading {config['file']}...", flush=True)
    file_path = DATA_DIR / config['file']
    df = _load_text_label_csv(file_path, max_rows=config['max_load'])
    print(f"  Loaded {len(df):,} rows with {df['author_id'].nunique():,} unique authors", flush=True)
    
    # Basic cleaning - vectorized
    print("Cleaning text...", flush=True)
    df['text'] = df['text'].astype(str).str.replace(r'http\S+', '[URL]', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['word_count'] = df['text'].str.split().str.len()
    
    # FIX #5: Lower minimum word count from 10 to 5
    df = df[df['word_count'] >= 5].copy()
    print(f"  After filtering (min 5 words): {len(df):,} rows", flush=True)
    
    # Decade grouping for birth_year
    if config['balance_strategy'] == 'decade_grouping':
        df['label_original'] = df['label']
        df['label'] = df['label'].apply(decade_label)
        df = df[df['label'].notna()].copy()
        print(f"  After decade grouping: {len(df):,} rows", flush=True)
        print(f"  Decade distribution: {df['label'].value_counts().to_dict()}", flush=True)
    
    # Balance by undersampling (per class)
    print("Balancing...", flush=True)
    target = config['target_per_class']
    balanced = []
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        if len(subset) > target:
            # Sample by author to maintain author integrity
            unique_authors = subset['author_id'].unique()
            if len(unique_authors) > target // 5:  # Assuming avg 5 posts per author
                sampled_authors = np.random.RandomState(RANDOM_SEED).choice(
                    unique_authors, 
                    size=min(len(unique_authors), target // 3),  # Sample authors
                    replace=False
                )
                subset = subset[subset['author_id'].isin(sampled_authors)]
            if len(subset) > target:
                subset = subset.sample(n=target, random_state=RANDOM_SEED)
        balanced.append(subset)
    df = pd.concat(balanced, ignore_index=True)
    # Ensure labels are consistently strings
    df['label'] = df['label'].astype(str)
    print(f"  Balanced: {len(df):,} rows", flush=True)
    print(f"  Distribution: {df['label'].value_counts().to_dict()}", flush=True)
    
    # Audit leakage (using expanded patterns)
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
    df_cleaned = df[['author_id', 'text_clean', 'label']].copy()
    df_cleaned.columns = ['author_id', 'text', 'label']
    
    # FIX #5: Lower threshold for cleaned text (3 words minimum)
    df_cleaned['word_count'] = df_cleaned['text'].str.split().str.len()
    df_cleaned = df_cleaned[df_cleaned['word_count'] >= 3].drop(columns=['word_count'])
    
    # Audit after
    leak_rate_after = df_cleaned.head(2000)['text'].astype(str).str.contains(LEAK_PATTERN, regex=True).mean() * 100
    print(f"  Leakage rate (after): {leak_rate_after:.1f}%", flush=True)
    
    # Align indices
    common = df_polluted.index.intersection(df_cleaned.index)
    df_polluted = df_polluted.loc[common, ['author_id', 'text', 'label']]
    df_cleaned = df_cleaned.loc[common]
    print(f"  Aligned samples: {len(df_polluted):,}", flush=True)
    
    # FIX #1: Author-level split
    print("Creating author-level splits...", flush=True)
    df_polluted, train_idx, dev_idx, test_idx = _author_level_train_dev_test_split(df_polluted)
    df_cleaned = df_cleaned.loc[df_polluted.index]
    
    # Verify no author overlap between splits
    train_authors = set(df_polluted.loc[train_idx, 'author_id'])
    dev_authors = set(df_polluted.loc[dev_idx, 'author_id'])
    test_authors = set(df_polluted.loc[test_idx, 'author_id'])
    
    train_dev_overlap = train_authors & dev_authors
    train_test_overlap = train_authors & test_authors
    dev_test_overlap = dev_authors & test_authors
    
    if train_dev_overlap or train_test_overlap or dev_test_overlap:
        print(f"  WARNING: Author overlap detected! This should not happen.")
    else:
        print(f"  ✓ No author overlap between splits (proper author-level split)")
    
    print(f"  Train: {len(train_idx)} samples ({len(train_authors)} authors)", flush=True)
    print(f"  Dev: {len(dev_idx)} samples ({len(dev_authors)} authors)", flush=True)
    print(f"  Test: {len(test_idx)} samples ({len(test_authors)} authors)", flush=True)
    
    # Save files (without author_id column for downstream use)
    print("Saving files...", flush=True)
    save_cols = ['text', 'label']
    
    # P0
    df_polluted[save_cols].to_csv(OUTPUT_DIR / f'P0_data_prep/{name}/polluted/full.csv', index=False)
    df_cleaned[save_cols].to_csv(OUTPUT_DIR / f'P0_data_prep/{name}/cleaned/full.csv', index=False)
    
    # Save cleaning validation sample (for manual review as per P0.2)
    validation_sample = df_cleaned.sample(min(200, len(df_cleaned)), random_state=RANDOM_SEED)
    validation_sample[save_cols].to_csv(OUTPUT_DIR / f'P0_data_prep/{name}/cleaning_validation_sample.csv', index=False)
    
    with open(OUTPUT_DIR / f'P0_data_prep/{name}/leakage_audit.json', 'w') as f:
        json.dump({
            'leak_rate_before': leak_rate_before, 
            'leak_rate_after': leak_rate_after,
            'total_samples': len(df_polluted),
            'unique_authors': df_polluted['author_id'].nunique(),
            'train_authors': len(train_authors),
            'dev_authors': len(dev_authors),
            'test_authors': len(test_authors),
        }, f, indent=2)
    
    # E1 - baselines (polluted)
    df_polluted.loc[train_idx, save_cols].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/train.csv', index=False)
    df_polluted.loc[dev_idx, save_cols].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/dev.csv', index=False)
    df_polluted.loc[test_idx, save_cols].to_csv(OUTPUT_DIR / f'E1_baselines/{name}/test.csv', index=False)
    
    # E2 - hypothesis (both)
    df_polluted.loc[train_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/train.csv', index=False)
    df_polluted.loc[dev_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/dev.csv', index=False)
    df_polluted.loc[test_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/polluted/test.csv', index=False)
    df_cleaned.loc[train_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/train.csv', index=False)
    df_cleaned.loc[dev_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/dev.csv', index=False)
    df_cleaned.loc[test_idx, save_cols].to_csv(OUTPUT_DIR / f'E2_hypothesis/{name}/cleaned/test.csv', index=False)
    
    # E4 - final test
    df_polluted.loc[test_idx, save_cols].to_csv(OUTPUT_DIR / f'E4_final/{name}/test_polluted.csv', index=False)
    df_cleaned.loc[test_idx, save_cols].to_csv(OUTPUT_DIR / f'E4_final/{name}/test_cleaned.csv', index=False)
    
    print(f"  Done saving {name}!", flush=True)
    
    return {
        'dataset': name,
        'total': len(df_polluted),
        'unique_authors': df_polluted['author_id'].nunique(),
        'train': len(train_idx),
        'dev': len(dev_idx),
        'test': len(test_idx),
        'leak_before': f"{leak_rate_before:.1f}%",
        'leak_after': f"{leak_rate_after:.1f}%",
    }


def main():
    print("="*50)
    print("REFINED DATASET PREPARATION")
    print("="*50)
    print("Fixes applied:")
    print("  1. Author-level splitting (prevents identity leakage)")
    print("  2. Expanded leakage patterns (husband/wife, pregnant, etc.)")
    print("  3. Less aggressive undersampling (more data)")
    print("  4. Balanced decade grouping")
    print("  5. Lower minimum word count")
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
    
    print("\nAll datasets prepared with refined methodology!")
    print(f"  Folder: {OUTPUT_DIR.absolute()}")
    print("\nKey improvements:")
    print("  - Author-level splits prevent identity leakage")
    print("  - More comprehensive leakage pattern detection")
    print("  - Larger sample sizes for better statistical power")
    print("  - Validation samples saved for manual review (P0.2)")


if __name__ == "__main__":
    main()
