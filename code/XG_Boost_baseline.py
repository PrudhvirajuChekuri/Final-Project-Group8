#%%
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import xgboost as xgb

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUBMISSION_PATH = 'submission_xgb.csv'

# 1. Load data
def load_data(train_path='data/train.csv', test_path='data/test.csv'):
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

# 3. Prepare features using char_wb n-grams
def prepare_features(train_texts, test_texts, max_features=10000):
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=max_features,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 4. Main pipeline
def main():
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

    # Train XGBoost model
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)

    # Validate
    y_val_pred = clf.predict(X_val)
    print("Validation Classification Report:\n")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print(f"Macro F1 on validation: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

    # Retrain on full data
    clf_full = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
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

if __name__ == '__main__':
    main()
# %%
