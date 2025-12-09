from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


DATA_DIR_CANDIDATES = [
    Path('..') / 'data',
]


def resolve_data_dir() -> Path:
    for p in DATA_DIR_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError('Could not find data directory')


def load_gender_dataset(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / 'gender.csv', header=None, names=['text', 'label'])
    df = df[df['label'].isin([0, 1, '0', '1'])].copy()
    df['label'] = df['label'].astype(int)
    df = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)
    return df


def remove_leak_tokens(text: str) -> str:
    tokens = str(text).split()
    cleaned = []
    for tok in tokens:
        low = tok.lower()
        if low in {'male', 'female', 'man', 'woman', 'guy', 'girl'}:
            continue
        has_digit = any(ch.isdigit() for ch in tok)
        has_gender_char = any(ch in 'MFmf' for ch in tok)
        if has_digit and has_gender_char:
            # Likely patterns like 18M, 21F, etc.
            continue
        cleaned.append(tok)
    return ' '.join(cleaned)


def run_experiment() -> None:
    data_dir = resolve_data_dir()
    print('Using data_dir:', data_dir.resolve())

    df = load_gender_dataset(data_dir)
    print('Dataset shape:', df.shape)
    print(df['label'].value_counts())

    df['text_clean'] = df['text'].apply(remove_leak_tokens)

    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
    split_idx = int(0.8 * len(df_shuffled))
    train_df = df_shuffled.iloc[:split_idx]
    test_df = df_shuffled.iloc[split_idx:]

    # Raw text model
    vec_raw = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
    X_train_raw = vec_raw.fit_transform(train_df['text'])
    X_test_raw = vec_raw.transform(test_df['text'])

    clf_raw = LinearSVC()
    clf_raw.fit(X_train_raw, train_df['label'])
    y_pred_raw = clf_raw.predict(X_test_raw)

    # Cleaned text model
    vec_clean = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
    X_train_clean = vec_clean.fit_transform(train_df['text_clean'])
    X_test_clean = vec_clean.transform(test_df['text_clean'])

    clf_clean = LinearSVC()
    clf_clean.fit(X_train_clean, train_df['label'])
    y_pred_clean = clf_clean.predict(X_test_clean)

    def show_scores(name: str, y_true, y_pred) -> None:
        print(name)
        print('  Accuracy:', round(accuracy_score(y_true, y_pred), 3))
        print('  Macro F1:', round(f1_score(y_true, y_pred, average='macro'), 3))

    print()
    print('TF-IDF + LinearSVC (gender)')
    show_scores('Original text:', test_df['label'], y_pred_raw)
    print()
    show_scores('Cleaned text:', test_df['label'], y_pred_clean)


if __name__ == '__main__':
    run_experiment()
