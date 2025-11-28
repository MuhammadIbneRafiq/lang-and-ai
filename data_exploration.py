"""
Data Exploration Script for Author Profiling Datasets
Explores Reddit author profiling data for stylometric analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DATA_DIR = Path("data")

def load_dataset(filename, sample_size=10000):
    """Load a dataset with optional sampling for large files."""
    filepath = DATA_DIR / filename
    print(f"\n{'='*60}")
    print(f"Loading: {filename}")
    print(f"File size: {filepath.stat().st_size / 1e6:.1f} MB")
    
    # Read full file to get counts, then sample
    df = pd.read_csv(filepath, header=None, names=['text', 'label'])
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    
    if total_rows > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to: {sample_size:,} rows")
    
    return df, total_rows

def basic_stats(df, name):
    """Print basic statistics for a dataset."""
    print(f"\n--- {name} Stats ---")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Label distribution:")
    print(df['label'].value_counts().head(10))
    
    # Text statistics
    df['text_length'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    print(f"\nText length stats:")
    print(f"  Mean chars: {df['text_length'].mean():.0f}")
    print(f"  Median chars: {df['text_length'].median():.0f}")
    print(f"  Mean words: {df['word_count'].mean():.0f}")
    print(f"  Median words: {df['word_count'].median():.0f}")
    
    return df

def detect_label_leaking_tokens(text):
    """Detect potential label-leaking tokens in text."""
    patterns = {
        'age_gender': r'\b(\d{1,2})\s*[MFmf]\b|\b[MFmf]\s*(\d{1,2})\b',  # "18F", "M25"
        'i_am_age': r"[Ii]'?m\s+(\d{1,2})\b|[Ii]\s+am\s+(\d{1,2})\b",  # "I'm 18", "I am 25"
        'i_am_gender': r"[Ii]'?m\s+a?\s*(male|female|man|woman|guy|girl)\b",
        'age_years_old': r'\b(\d{1,2})\s*(?:years?\s*old|yo|y\.o\.)\b',
        'born_in': r'\bborn\s+in\s+(\d{4})\b',
    }
    
    found = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, str(text), re.IGNORECASE)
        if matches:
            found[name] = matches
    return found

def analyze_label_leaking(df, label_type):
    """Analyze prevalence of label-leaking tokens."""
    print(f"\n--- Label Leaking Analysis for {label_type} ---")
    
    leaking_count = 0
    sample = df.head(1000)  # Check first 1000
    
    for _, row in sample.iterrows():
        leaks = detect_label_leaking_tokens(row['text'])
        if leaks:
            leaking_count += 1
    
    pct = (leaking_count / len(sample)) * 100
    print(f"Posts with potential label-leaking tokens: {leaking_count}/{len(sample)} ({pct:.1f}%)")
    
    return leaking_count

def plot_label_distributions(datasets_info):
    """Create distribution plots for all datasets."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (name, df, _) in enumerate(datasets_info):
        ax = axes[idx]
        
        if name in ['birth_year']:
            # Numeric - histogram
            df['label'].astype(int).hist(ax=ax, bins=30, edgecolor='black')
            ax.set_xlabel('Birth Year')
        elif name in ['gender', 'extrovert_introvert', 'feeling_thinking', 
                      'judging_perceiving', 'sensing_intuitive']:
            # Binary - bar chart
            df['label'].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_xlabel('Label')
        else:
            # Categorical - top 10 bar chart
            df['label'].value_counts().head(10).plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_xlabel('Label (Top 10)')
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_ylabel('Count')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('label_distributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved: label_distributions.png")
    plt.close()

def plot_text_length_distributions(datasets_info):
    """Plot text length distributions across datasets."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (name, df, _) in enumerate(datasets_info):
        ax = axes[idx]
        
        # Log scale for better visualization
        word_counts = df['text'].astype(str).apply(lambda x: len(x.split()))
        word_counts = word_counts[word_counts > 0]  # Remove zeros
        
        ax.hist(np.log10(word_counts + 1), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Log10(Word Count)')
        ax.set_ylabel('Frequency')
        ax.set_title(name.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig('text_length_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: text_length_distributions.png")
    plt.close()

def plot_dataset_overview(datasets_info):
    """Create overview comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [info[0] for info in datasets_info]
    total_rows = [info[2] for info in datasets_info]
    unique_labels = [info[1]['label'].nunique() for info in datasets_info]
    avg_words = [info[1]['text'].astype(str).apply(lambda x: len(x.split())).mean() 
                 for info in datasets_info]
    
    # Total samples
    ax = axes[0]
    bars = ax.bar(range(len(names)), total_rows, color=sns.color_palette("husl", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, fontsize=8)
    ax.set_ylabel('Total Samples')
    ax.set_title('Dataset Sizes')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Unique labels
    ax = axes[1]
    bars = ax.bar(range(len(names)), unique_labels, color=sns.color_palette("husl", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, fontsize=8)
    ax.set_ylabel('Unique Labels')
    ax.set_title('Label Cardinality')
    
    # Average word count
    ax = axes[2]
    bars = ax.bar(range(len(names)), avg_words, color=sns.color_palette("husl", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, fontsize=8)
    ax.set_ylabel('Average Words per Sample')
    ax.set_title('Text Length')
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=150, bbox_inches='tight')
    print("Saved: dataset_overview.png")
    plt.close()

def clean_text(text):
    """Basic text cleaning."""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', 'url', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very long repeated characters
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    return text.strip()

def create_clean_sample(df, name, n=5000):
    """Create a cleaned sample dataset for experiments."""
    sample = df.sample(n=min(n, len(df)), random_state=42).copy()
    sample['text_clean'] = sample['text'].apply(clean_text)
    sample['word_count'] = sample['text_clean'].apply(lambda x: len(x.split()))
    
    # Filter out very short or very long texts
    sample = sample[(sample['word_count'] >= 10) & (sample['word_count'] <= 5000)]
    
    output_path = DATA_DIR / f"{name}_clean_sample.csv"
    sample[['text_clean', 'label']].to_csv(output_path, index=False)
    print(f"Saved clean sample: {output_path} ({len(sample)} rows)")
    
    return sample

def main():
    print("="*60)
    print("REDDIT AUTHOR PROFILING DATA EXPLORATION")
    print("="*60)
    
    # List all datasets
    files = [
        'gender.csv',
        'birth_year.csv', 
        'political_leaning.csv',
        'nationality.csv',
        'extrovert_introvert.csv',
        'feeling_thinking.csv',
        'judging_perceiving.csv',
        'sensing_intuitive.csv',
    ]
    
    datasets_info = []
    
    for filename in files:
        name = filename.replace('.csv', '')
        df, total_rows = load_dataset(filename, sample_size=5000)
        df = basic_stats(df, name)
        
        # Check for label leaking in relevant datasets
        if name in ['gender', 'birth_year']:
            analyze_label_leaking(df, name)
        
        datasets_info.append((name, df, total_rows))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_dataset_overview(datasets_info)
    plot_label_distributions(datasets_info)
    plot_text_length_distributions(datasets_info)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    summary_data = []
    for name, df, total in datasets_info:
        word_counts = df['text'].astype(str).apply(lambda x: len(x.split()))
        summary_data.append({
            'Dataset': name,
            'Total Samples': f"{total:,}",
            'Unique Labels': df['label'].nunique(),
            'Avg Words': f"{word_counts.mean():.0f}",
            'Median Words': f"{word_counts.median():.0f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('dataset_summary.csv', index=False)
    print("\nSaved: dataset_summary.csv")
    
    # Create clean samples for key datasets (for your proposals)
    print("\n" + "="*60)
    print("CREATING CLEAN SAMPLES FOR EXPERIMENTS")
    print("="*60)
    
    key_datasets = ['gender', 'birth_year', 'political_leaning']
    for name, df, _ in datasets_info:
        if name in key_datasets:
            create_clean_sample(df, name)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR PROPOSALS")
    print("="*60)
    
    print("""
    PROPOSAL 2 (Stylometric Features):
    - Use: gender.csv, birth_year.csv 
    - These have clear binary/ordinal labels for age/gender profiling
    - Combine with subreddit info if available for cross-domain analysis
    
    PROPOSAL 3 (Label-Leaking Tokens):
    - Use: gender.csv, birth_year.csv
    - High prevalence of explicit mentions like "I'm 18F", "21M", etc.
    - Perfect for studying shortcut learning
    
    KEY DATASETS:
    1. gender.csv - Binary classification (0/1)
    2. birth_year.csv - Ordinal/regression (birth years)
    3. political_leaning.csv - Binary (left/right)
    4. MBTI files - Binary personality dimensions
    5. nationality.csv - Multi-class (many countries)
    """)

if __name__ == "__main__":
    main()
