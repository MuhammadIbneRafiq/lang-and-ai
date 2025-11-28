"""
Comprehensive Analysis Script for Author Profiling Data
Generates all visualizations for Proposals 2 & 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

DATA_DIR = Path('data')

# Load key datasets
print('Loading datasets...')
gender_df = pd.read_csv(DATA_DIR / 'gender.csv', header=None, names=['text', 'label'])
# Clean gender labels - keep only 0 and 1
gender_df = gender_df[gender_df['label'].isin([0, 1, '0', '1'])]
gender_df['label'] = gender_df['label'].astype(int)
gender_df = gender_df.sample(n=min(10000, len(gender_df)), random_state=42)
print(f'Gender labels: {gender_df["label"].value_counts().to_dict()}')

age_df = pd.read_csv(DATA_DIR / 'birth_year.csv', header=None, names=['text', 'label'])
# Clean age labels - keep only valid birth years
age_df['label'] = pd.to_numeric(age_df['label'], errors='coerce')
age_df = age_df.dropna(subset=['label'])
age_df['label'] = age_df['label'].astype(int)
age_df = age_df[(age_df['label'] >= 1940) & (age_df['label'] <= 2010)]  # Reasonable birth years
age_df = age_df.sample(n=min(10000, len(age_df)), random_state=42)
print(f'Age range: {age_df["label"].min()} - {age_df["label"].max()}')

pol_df = pd.read_csv(DATA_DIR / 'political_leaning.csv', header=None, names=['text', 'label'])
pol_df = pol_df[pol_df['label'].isin(['left', 'right', 'center'])]
pol_df = pol_df.sample(n=min(10000, len(pol_df)), random_state=42)
print(f'Political labels: {pol_df["label"].value_counts().to_dict()}')

# Label leaking patterns
LEAK_PATTERNS = {
    'age_gender_combo': r'\b(\d{1,2})\s*[MFmf]\b|\b[MFmf]\s*(\d{1,2})\b',
    'i_am_age': r"[Ii]'?m\s+(\d{1,2})\b|[Ii]\s+am\s+(\d{1,2})\b",
    'i_am_gender': r"[Ii]'?m\s+a?\s*(male|female|man|woman|guy|girl)\b",
    'age_years_old': r'\b(\d{1,2})\s*(?:years?\s*old|yo|y\.o\.)\b',
    'age_brackets': r'\(\s*(\d{1,2})\s*[MFmf]\s*\)|\(\s*[MFmf]\s*(\d{1,2})\s*\)',
}

def has_leak(text):
    for pattern in LEAK_PATTERNS.values():
        if re.search(pattern, str(text), re.IGNORECASE):
            return True
    return False

print('Analyzing label leaking...')
gender_df['has_leak'] = gender_df['text'].apply(has_leak)
age_df['has_leak'] = age_df['text'].apply(has_leak)

# Create key distributions plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
gender_counts = gender_df['label'].value_counts()
gender_counts.index = ['Male (0)' if x == 0 else 'Female (1)' for x in gender_counts.index]
gender_counts.plot(kind='bar', ax=ax, color=['steelblue', 'coral'], edgecolor='black')
ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)

ax = axes[1]
age_df['label'].astype(int).hist(ax=ax, bins=40, color='seagreen', edgecolor='black')
ax.set_title('Birth Year Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Birth Year')
ax.set_ylabel('Count')

ax = axes[2]
pol_counts = pol_df['label'].value_counts()
colors = {'left': 'blue', 'center': 'gray', 'right': 'red'}
pol_counts.plot(kind='bar', ax=ax, color=[colors.get(x, 'gray') for x in pol_counts.index], edgecolor='black')
ax.set_title('Political Leaning Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('key_dataset_distributions.png', dpi=150, bbox_inches='tight')
print('Saved: key_dataset_distributions.png')

# Label leaking analysis plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
leak_by_gender = gender_df.groupby('label')['has_leak'].mean() * 100
label_names = {0: 'Male', 1: 'Female'}
leak_by_gender.index = [label_names.get(x, str(x)) for x in leak_by_gender.index]
colors_list = ['steelblue' if 'Male' in str(x) else 'coral' for x in leak_by_gender.index]
bars = ax.bar(leak_by_gender.index, leak_by_gender.values, color=colors_list, edgecolor='black')
ax.set_ylabel('% with Label-Leaking Tokens')
ax.set_title('Label Leaking by Gender', fontsize=12, fontweight='bold')
for bar, val in zip(bars, leak_by_gender.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(leak_by_gender) * 1.2)

ax = axes[1]
datasets = ['Gender', 'Birth Year']
leak_rates = [gender_df['has_leak'].mean() * 100, age_df['has_leak'].mean() * 100]
bars = ax.bar(datasets, leak_rates, color=['mediumpurple', 'seagreen'], edgecolor='black')
ax.set_ylabel('% with Label-Leaking Tokens')
ax.set_title('Label Leaking Prevalence by Dataset', fontsize=12, fontweight='bold')
for bar, val in zip(bars, leak_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(leak_rates) * 1.2)

plt.tight_layout()
plt.savefig('label_leaking_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: label_leaking_analysis.png')

print(f'\nGender dataset: {gender_df["has_leak"].mean()*100:.1f}% samples have label-leaking tokens')
print(f'Birth Year dataset: {age_df["has_leak"].mean()*100:.1f}% samples have label-leaking tokens')

# Stylometric features
def extract_stylometric_features(text):
    text = str(text)
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = {}
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['sentence_count'] = max(len(sentences), 1)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    unique_words = set(w.lower() for w in words)
    features['vocab_richness'] = len(unique_words) / max(len(words), 1)
    
    punct_counts = Counter(c for c in text if c in string.punctuation)
    total_punct = sum(punct_counts.values())
    features['punct_ratio'] = total_punct / max(len(text), 1)
    features['exclamation_ratio'] = punct_counts.get('!', 0) / max(total_punct, 1)
    features['question_ratio'] = punct_counts.get('?', 0) / max(total_punct, 1)
    features['comma_ratio'] = punct_counts.get(',', 0) / max(total_punct, 1)
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'}
    lower_words = [w.lower() for w in words]
    features['function_word_ratio'] = sum(1 for w in lower_words if w in function_words) / max(len(words), 1)
    
    first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'}
    second_person = {'you', 'your', 'yours', 'yourself'}
    features['first_person_ratio'] = sum(1 for w in lower_words if w in first_person) / max(len(words), 1)
    features['second_person_ratio'] = sum(1 for w in lower_words if w in second_person) / max(len(words), 1)
    
    return features

print('\nExtracting stylometric features...')
gender_features = pd.DataFrame([extract_stylometric_features(t) for t in gender_df['text']])
gender_features['label'] = gender_df['label'].values
feature_cols = [c for c in gender_features.columns if c != 'label']

# Feature distributions by gender
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()

for idx, feat in enumerate(feature_cols):
    ax = axes[idx]
    male_vals = gender_features[gender_features['label'] == 0][feat]
    female_vals = gender_features[gender_features['label'] == 1][feat]
    ax.hist(male_vals, bins=30, alpha=0.6, label='Male', color='steelblue', density=True)
    ax.hist(female_vals, bins=30, alpha=0.6, label='Female', color='coral', density=True)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
    ax.legend(fontsize=8)

for idx in range(len(feature_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('stylometric_features_by_gender.png', dpi=150, bbox_inches='tight')
print('Saved: stylometric_features_by_gender.png')

# Feature comparison table
print('\n' + '='*70)
print('STYLOMETRIC FEATURE COMPARISON BY GENDER')
print('='*70)
print(f'{"Feature":<25} {"Male Mean":>12} {"Female Mean":>12} {"Diff %":>10} {"p-value":>12}')
print('-'*70)

for feat in feature_cols:
    male = gender_features[gender_features['label'] == 0][feat]
    female = gender_features[gender_features['label'] == 1][feat]
    male_mean = male.mean()
    female_mean = female.mean()
    diff_pct = ((female_mean - male_mean) / male_mean * 100) if male_mean != 0 else 0
    t_stat, p_val = stats.ttest_ind(male, female)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f'{feat:<25} {male_mean:>12.4f} {female_mean:>12.4f} {diff_pct:>+9.1f}% {p_val:>10.2e} {sig}')

# Baseline models
print('\n' + '='*60)
print('BASELINE: Stylometric Features for Gender Prediction')
print('='*60)

X = gender_features[feature_cols].values
y = gender_features['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print(f'Logistic Regression: Acc={accuracy_score(y_test, lr_pred):.3f}, F1={f1_score(y_test, lr_pred, average="macro"):.3f}')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f'Random Forest: Acc={accuracy_score(y_test, rf_pred):.3f}, F1={f1_score(y_test, rf_pred, average="macro"):.3f}')

# Feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
coef_df = pd.DataFrame({'feature': feature_cols, 'coef': np.abs(lr.coef_[0])})
coef_df = coef_df.sort_values('coef', ascending=True)
ax.barh(coef_df['feature'], coef_df['coef'], color='steelblue', edgecolor='black')
ax.set_xlabel('|Coefficient|')
ax.set_title('Logistic Regression Feature Importance', fontsize=12, fontweight='bold')

ax = axes[1]
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=True)
ax.barh(imp_df['feature'], imp_df['importance'], color='seagreen', edgecolor='black')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print('Saved: feature_importance.png')

# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = gender_features[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Stylometric Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
print('Saved: feature_correlation_matrix.png')

# Save analysis data
gender_df.to_csv('data/gender_analysis_ready.csv', index=False)
gender_features.to_csv('data/gender_stylometric_features.csv', index=False)
print('\nSaved analysis data to data/ directory')

print('\n' + '='*60)
print('ANALYSIS COMPLETE!')
print('='*60)
