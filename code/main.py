# %% [markdown]
# # Imports
# Import all necessary libraries for data handling, modeling, augmentation, and evaluation.

# %%
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset
from textattack.augmentation import Augmenter
from textattack.transformations import (
    WordSwapRandomCharacterDeletion,
    WordSwapChangeLocation,
    CompositeTransformation
)

# %% [markdown]
# # Configuration
# Set model parameters like model name, max token length, batch size, and random seed.

# %%
MODEL_NAME = "tbs17/MathBERT"
MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 42

# %% [markdown]
# # Data Augmentation Class
# Defines `MathAugmenter` class using TextAttack for simple text augmentations.

# %%
class MathAugmenter:
    def __init__(self, num_augments=2):
        self.num_augments = num_augments
        transformation = CompositeTransformation([
            WordSwapRandomCharacterDeletion(),
            WordSwapChangeLocation()
        ])
        self.augmenter = Augmenter(transformation=transformation)

    def augment(self, text):
        augmented_texts = self.augmenter.augment(text, num_augmented_outputs=self.num_augments)
        return augmented_texts

# %% [markdown]
# # Dataset Class
# Prepares tokenized dataset for use with Hugging Face's Trainer API.

# %%
class MathDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# %% [markdown]
# # Dataset Processor Class
# Loads training and test data from CSV files.

# %%
class DatasetProcessor:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

    def get_data(self):
        return self.train_df, self.test_df

# %% [markdown]
# # Model Trainer Class
# Handles tokenization, model initialization, training, and prediction using Hugging Face Transformers.

# %%
class ModelTrainer:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)

    def train(self, train_texts, train_labels):
        dataset = MathDataset(train_texts, train_labels, self.tokenizer)
        args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="no",
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=3,
            seed=SEED,
            logging_steps=10,
            save_strategy="no"
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
        return trainer

    def predict(self, trainer, texts):
        dataset = MathDataset(texts, [0] * len(texts), self.tokenizer)
        preds = trainer.predict(dataset)
        return np.argmax(preds.predictions, axis=1)

# %% [markdown]
# # Main Pipeline
# 1. Loads data
# 2. Applies augmentation
# 3. Trains the model
# 4. Predicts on the test set

# %%
def main():
    processor = DatasetProcessor("data/train.csv", "data/test.csv")
    train_df, test_df = processor.get_data()

    augmenter = MathAugmenter()
    train_df['augmented'] = train_df['Question'].apply(lambda x: augmenter.augment(x)[0])
    train_texts = train_df['augmented'].tolist()
    train_labels = train_df['label'].tolist()

    trainer_obj = ModelTrainer()
    trainer = trainer_obj.train(train_texts, train_labels)

    test_texts = test_df['Question'].tolist()
    predictions = trainer_obj.predict(trainer, test_texts)
    print(predictions)

# %% [markdown]
# # Run the Main Function

# %%
if __name__ == "__main__":
    main()


