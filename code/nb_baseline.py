#%%
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import time

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 3
SUBMISSION_PATH = 'submission_nb.csv'

# 1. Load data
def load_data(train_path='/home/ubuntu/github_NLP/code/data/train.csv', test_path='/home/ubuntu/github_NLP/code/data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df

# 2. Simple text cleaning (optional math placeholder)
def clean_text(text):
    # Replace math spans with placeholder
    text = re.sub(r"\$(.*?)\$", ' MATH ', text)
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^a-z0-9 ]", ' ', text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# 3. Prepare features
def prepare_features(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 4. Main pipeline
def main():
    # Load
    start_time = time.time()
    train_df, test_df = load_data()

    # Clean text
    train_texts = train_df['Question'].apply(clean_text).tolist()
    test_texts  = test_df['Question'].apply(clean_text).tolist()
    y = train_df['label'].values

    # Feature extraction
    X_train, X_test, vect = prepare_features(train_texts, test_texts)

    # Optional: validation split for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_tr, y_tr)

    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    print("Validation Classification Report:\n")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print(f"Micro F1 on validation: {f1_score(y_val, y_val_pred, average='micro'):.4f}")

    # Retrain on full training data
    clf_full = MultinomialNB()
    clf_full.fit(X_train, y)

    # Predict on test set
    test_preds = clf_full.predict(X_test)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'label': test_preds
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

# %%
