# %% [markdown]
# # Import Libraries

# %%
import re
import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


#%%
import warnings

# Suppress specific sklearn warning about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

# %% [markdown]
# # Load Data

# %%
train = pd.read_csv("/home/ubuntu/github_NLP/code/data/train.csv")
test = pd.read_csv("/home/ubuntu/github_NLP/code/data/test.csv")

#%%
# Create output directory if it doesn't exist
output_dir = "/home/ubuntu/github_NLP/code/output/models/lightGBM/"
os.makedirs(output_dir, exist_ok=True)

# %%
train["label"].value_counts()


# %%
X = train["Question"]
y = train["label"]

# %% [markdown]
# # Split data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# # Create Training Pipeline with LightGBM

# %%
lgbm = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LGBMClassifier(is_unbalance = True, verbose = -1))
              ])

lgbm.fit(X_train, y_train)

#%%
# Save the trained LightGBM pipeline
model_path = os.path.join(output_dir, "lgbm_pipeline.joblib")
joblib.dump(lgbm, model_path)
print(f"LightGBM model saved to {model_path}")

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

# Make predictions on the test dataset
predicted_labels = lgbm.predict(test['Question'])

submission_df = pd.DataFrame({
    'id': test['id'],
    'label': predicted_labels
})


submission_filename = 'submission_lgbm.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file '{submission_filename}' created successfully.")


