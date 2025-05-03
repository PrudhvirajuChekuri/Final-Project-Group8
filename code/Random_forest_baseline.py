#%%
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUBMISSION_PATH = 'submission_rf.csv'

# 1. Load data
def load_data(train_path='/home/ubuntu/github_NLP/code/data/train.csv', test_path='/home/ubuntu/github_NLP/code/data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df

# 2. Simple text cleaning with math placeholder
def clean_text(text):
    text = re.sub(r"\$(.*?)\$", ' MATH ', text)  # Replace math with placeholder
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", ' ', text)        # Remove special chars
    text = re.sub(r"\s+", ' ', text).strip()      # Normalize whitespace
    return text

# 3. Prepare features
def prepare_features(train_texts, test_texts, max_features=7000):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 4. Main pipeline
def main():
    start_time = time.time()
    train_df, test_df = load_data()

    # Preprocess text
    train_texts = train_df['Question'].apply(clean_text).tolist()
    test_texts  = test_df['Question'].apply(clean_text).tolist()
    y = train_df['label'].values

    # TF-IDF features
    X_train, X_test, vect = prepare_features(train_texts, test_texts)

    # Split validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    # Validate
    y_val_pred = clf.predict(X_val)
    print("Validation Classification Report:\n")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print(f"Micro F1 on validation: {f1_score(y_val, y_val_pred, average='micro'):.4f}")

    # Retrain on full data
    clf_full = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf_full.fit(X_train, y)

    # Predict on test set
    test_preds = clf_full.predict(X_test)

    # Submission file
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
