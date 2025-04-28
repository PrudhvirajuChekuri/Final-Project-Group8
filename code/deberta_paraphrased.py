# %% [markdown]
# # Install Packages

# %%
!pip install evaluate -q

# %% [markdown]
# # Import Libraries

# %%
import evaluate
import pandas as pd
import numpy as np
import torch
import re
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# %% [markdown]
# # Set Device

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Config

# %%
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
NUM_LABELS = 8
TEST_SIZE = 0.1
VALID_SIZE = 0.1
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 10
OUTPUT_DIR = "./math_classifier_results"
LOGGING_DIR = "./math_classifier_logs"

# %% [markdown]
# # Load Data

# %%
df = pd.read_csv("/kaggle/input/ka-challenges-data-pp/train_pp.csv")
df = df.dropna().reset_index(drop=True)
df.drop("Question", axis=1, inplace=True)
df.rename(columns={'Question_pp': 'Question'}, inplace=True)

print(f"Loaded DataFrame shape: {df.shape}")
print("Label distribution:\n", df['label'].value_counts())

# %% [markdown]
# # Data Preprocessing

# %%
def clean_math_text_final(text):
    # Skipping regex preprocessing since the questions are already paraphrased
    return text


print("\n--- Applying Text Cleaning ---")
df['cleaned_question'] = df['Question'].apply(clean_math_text_final)
print("Cleaning applied.")

# %% [markdown]
# # Prepare Data for Training

# %%
print("\n--- Creating Dataset & Casting Label Type ---")
dataset = Dataset.from_pandas(df)

class_label_feature = ClassLabel(num_classes=NUM_LABELS)
dataset = dataset.cast_column('label', class_label_feature)

print(f"Dataset features after casting 'label' column:")
print(dataset.features)

print("\n--- Splitting Dataset ---")
train_test_split = dataset.train_test_split(test_size=TEST_SIZE, stratify_by_column='label')
train_valid_split = train_test_split['train'].train_test_split(test_size=VALID_SIZE / (1 - TEST_SIZE), stratify_by_column='label')

final_datasets = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['test'],
    'test': train_test_split['test']
})
print("Dataset splits created:")
print(final_datasets)

# %% [markdown]
# # Tokenization

# %%
print("\n--- Tokenization ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    
    return tokenizer(examples["cleaned_question"],
                     padding="max_length",
                     truncation=True,
                     max_length=MAX_LENGTH)

tokenized_datasets = final_datasets.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["Question", "cleaned_question"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print("Tokenization complete.")
print("Example train sample:", tokenized_datasets["train"][0])

# %% [markdown]
# # Define Evaluation Metrics

# %%
print("\n--- Setting up Metrics ---")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    f1_micro = f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
    return {"accuracy": accuracy, "f1_weighted": f1_weighted, "f1_micro": f1_micro}

# %% [markdown]
# # Train Deberta model on Paraphrased Math Problems

# %%
print("\n--- Loading Model ---")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model.to(device)
print(f"Model '{MODEL_NAME}' loaded for {NUM_LABELS}-class classification.")

print("\n--- Defining Training Arguments ---")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOGGING_DIR,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none",
    fp16=True,
    save_total_limit=3
)

print("\n--- Initializing Trainer ---")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n--- Starting Fine-Tuning ---")
train_result = trainer.train()

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


trainer.save_model(OUTPUT_DIR + "/best_model")
print("Training finished. Best model saved.")


print("\n--- Evaluating on Test Set ---")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

print("Test Set Evaluation Results:")
print(test_results)
trainer.log_metrics("eval", test_results)
trainer.save_metrics("eval", test_results)

# %% [markdown]
# # Run Predictions on Test data and make Submission

# %%
print("\n--- Preparing Competition Test Set ---")

comp_test_df = pd.read_csv("/kaggle/input/ka-challenges-data-pp/test_pp.csv")
comp_test_df["Question_pp"] = comp_test_df["Question_pp"].fillna(comp_test_df["Question_pp"][0])
comp_test_df.drop("Question", axis=1, inplace=True)
comp_test_df.rename(columns={'Question_pp': 'Question'}, inplace=True)

print(f"Loaded competition test data shape: {comp_test_df.shape}")

print("Cleaning competition test data...")
comp_test_df['cleaned_question'] = comp_test_df['Question'].apply(clean_math_text_final)
print("Cleaning complete.")

predict_dataset = Dataset.from_pandas(comp_test_df[['cleaned_question']])
print("Competition test data converted to Dataset format.")
print(predict_dataset)

def tokenize_for_predict(examples):
    return tokenizer(examples["cleaned_question"],
                     padding="max_length",
                     truncation=True,
                     max_length=MAX_LENGTH)

print("\n--- Tokenizing Competition Test Set ---")
tokenized_predict_dataset = predict_dataset.map(tokenize_for_predict, batched=True)

tokenized_predict_dataset = tokenized_predict_dataset.remove_columns(["cleaned_question"])
tokenized_predict_dataset.set_format("torch")
print("Tokenization complete.")
print("Example prediction sample:", tokenized_predict_dataset[0])


print("\n--- Making Predictions ---")
predictions_output = trainer.predict(tokenized_predict_dataset)

logits = predictions_output.predictions

predicted_labels = np.argmax(logits, axis=-1)
print("Predictions generated.")
print("Example predicted labels:", predicted_labels[:10])

print("\n--- Creating Submission File ---")
submission_df = pd.DataFrame({
    'id': comp_test_df['id'],
    'label': predicted_labels
})

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file '{submission_filename}' created successfully.")
print(submission_df.head())


