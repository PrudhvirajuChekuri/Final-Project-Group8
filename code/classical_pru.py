# %% [markdown]
# # Import Libraries

# %%
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# %% [markdown]
# # Load Data

# %%
train = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
sample_sub = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")

display(train.shape, test.shape)
display(sample_sub.head())

# %%
train.head()

# %%
test.head()

# %%
train["label"].value_counts()

# %%
for ind, i in enumerate(train["Question"][80:100]):
    print("Q:", ind)
    print(i)
    print("\n")

# %% [markdown]
# - Many problems starts with random things in different forms in many examples like
#     - numbers
#     - Example ...
#     - task ...
# - Has links, random names
# - Has names of question creators, authors etc
# - Maybe extracted or scraped questions and are not derived by humans. Or these things are introduced randomly for competition purposes.

# %%
X = train["Question"]
y = train["label"]
display(X.shape, y.shape, X.head(2), y.head(2))

# %% [markdown]
# # Split data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# # Create Training Pipeline with Logistic Regression

# %%
lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(class_weight="balanced", multi_class='ovr'))
              ])
lr.fit(X_train, y_train)

# %% [markdown]
# # Make Predictions

# %%
train_preds_lr = lr.predict(X_train)
test_preds_lr = lr.predict(X_test)

# %% [markdown]
# # Evaluate

# %%
print("Logistic Regression Train Accuracy:", lr.score(X_train, y_train))
print("Logistic Regression Test Accuracy:", lr.score(X_test, y_test))

# %% [markdown]
# # Create Training Pipeline with LightGBM

# %%
lgbm = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LGBMClassifier(is_unbalance = True, verbose = -1))
              ])

lgbm.fit(X_train, y_train)

# %% [markdown]
# # Make Predictions

# %%
train_preds_lgbm = lgbm.predict(X_train)
test_preds_lgbm = lgbm.predict(X_test)

# %% [markdown]
# # Evaluation

# %%
print("LGBM Train Accuracy:", lgbm.score(X_train, y_train))
print("LGBM Test Accuracy:", lgbm.score(X_test, y_test))

# %% [markdown]
# # Data Preprocessing

# %%
def clean_math_text_final(text):
    text = str(text)
    text = re.sub(r'^\s*\d+\.\s*', '', text)    
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text

# %% [markdown]
# # Re-run both models on Cleaned Data

# %%
train['Question'] = train['Question'].apply(clean_math_text_final)
test['Question'] = test['Question'].apply(clean_math_text_final)

train.head()

# %%
X = train["Question"]
y = train["label"]
display(X.shape, y.shape, X.head(2), y.head(2))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(class_weight="balanced", multi_class='ovr'))
              ])
lr.fit(X_train, y_train)

# %%
train_preds_lr = lr.predict(X_train)
test_preds_lr = lr.predict(X_test)

# %%
print("Logistic Regression Train Accuracy:", lr.score(X_train, y_train))
print("Logistic Regression Test Accuracy:", lr.score(X_test, y_test))

# %%
lgbm = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LGBMClassifier(is_unbalance = True, verbose = -1))
              ])

lgbm.fit(X_train, y_train)

# %%
train_preds_lgbm = lgbm.predict(X_train)
test_preds_lgbm = lgbm.predict(X_test)

# %%
print("LGBM Train Accuracy:", lgbm.score(X_train, y_train))
print("LGBM Test Accuracy:", lgbm.score(X_test, y_test))

# %% [markdown]
# # Kaggle Competition Submission

# %%
predicted_labels = lgbm.predict(test["Question"])

submission_df = pd.DataFrame({
    'id': test['id'],
    'label': predicted_labels
})


submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file '{submission_filename}' created successfully.")
display(submission_df.head())


