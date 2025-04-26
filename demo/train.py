import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
from scipy.stats import mode
import sys

# Add the demo directory to the path so we can import from model_utils
sys.path.append(os.path.dirname(__file__))
from model_utils import save_model, preprocess_text

# Constants
MODEL_NAME = "tbs17/MathBERT"
MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 42
OUTPUT_DIR = "../code/output"

class MathDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def load_data(train_path, test_path):
    """
    Load and preprocess the data
    
    Args:
        train_path (str): Path to the training data
        test_path (str): Path to the test data
        
    Returns:
        tuple: (train_df, test_df)
    """
    print(f"Loading data from {train_path} and {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Display class distribution
    print("\nClass distribution in training data:")
    print(train_df['label'].value_counts().sort_index())
    
    # Clean and preprocess the text
    print("\nPreprocessing text...")
    train_df['cleaned'] = train_df['Question'].apply(preprocess_text)
    test_df['cleaned'] = test_df['Question'].apply(preprocess_text)
    
    return train_df, test_df

# def make_model_contiguous(model):
#     """
#     Make all parameters in the model contiguous to avoid issues when saving
#
#     Args:
#         model: The PyTorch model
#
#     Returns:
#         model: The model with contiguous parameters
#     """
#     for param_name, param in model.named_parameters():
#         if not param.is_contiguous():
#             model.state_dict()[param_name] = param.contiguous()
#     return model

def make_model_contiguous(model):
    """
    Make all parameters in the model own their memory (be contiguous)
    """
    for name, param in model.named_parameters():
        if not param.data.is_contiguous():
            # overwrite the tensor in-place with a contiguous copy
            param.data = param.data.contiguous()
    return model

def train_mathbert(train_df, test_df, n_splits=3, epochs=5):
    """
    Train the MathBERT model using cross-validation

    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Test data
        n_splits (int): Number of cross-validation folds
        epochs (int): Number of training epochs

    Returns:
        tuple: (best_model, tokenizer, final_predictions)
    """
    # Initialize tokenizer with math special tokens
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[MATH]']}) #

    # Prepare test dataset
    test_dataset = MathDataset(
        test_df['cleaned'].tolist(),
        [0]*len(test_df),
        tokenizer
    ) #

    # Cross-validation setup
    print(f"\nSetting up {n_splits}-fold cross-validation...") #
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED) #
    all_preds = []
    best_model = None
    best_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(
        train_df['cleaned'], train_df['label']
    )): #
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/{n_splits}") #
        print(f"{'='*50}")

        # Model initialization
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=8,
            ignore_mismatched_sizes=True
        ) #
        model.resize_token_embeddings(len(tokenizer)) #

        # *** Start Change ***
        # Ensure model parameters are contiguous *before* training starts
        print("\nMaking model parameters contiguous before training...") #
        model = make_model_contiguous(model) # Call the function here
        # *** End Change ***

        # Training arguments
        args = TrainingArguments(
            num_train_epochs=epochs,
            output_dir=f'./fold_{fold}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE*2,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=1,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            logging_dir='./logs',
            logging_steps=100,
            report_to='none',
            warmup_ratio=0.1,
            weight_decay=0.01,
            seed=SEED,
            load_best_model_at_end=True,
            metric_for_best_model='f1_micro'
        ) #

        # Trainer setup
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=MathDataset(
                train_df.iloc[train_idx]['cleaned'].tolist(),
                train_df.iloc[train_idx]['label'].values,
                tokenizer
            ), #
            eval_dataset=MathDataset(
                train_df.iloc[val_idx]['cleaned'].tolist(),
                train_df.iloc[val_idx]['label'].values,
                tokenizer
            ), #
            compute_metrics=lambda p: {
                'f1_micro': f1_score(p.label_ids, p.predictions.argmax(-1), average='micro')
            } #
        )

        # Training
        print("\nTraining model...") #
        trainer.train()

        # Evaluation
        print("\nEvaluating model on validation set...") #
        eval_result = trainer.evaluate() #
        print(f"Validation F1-micro: {eval_result['eval_f1_micro']:.4f}") #

        # Update the best model based on performance *after* training the fold
        current_f1 = eval_result['eval_f1_micro']
        if current_f1 > best_f1:
            best_f1 = current_f1
            # Since load_best_model_at_end=True, trainer.model is the best model for this fold
            # We need to ensure it's on CPU and contiguous before storing
            best_model_state_dict = {k: v.cpu().contiguous() for k, v in trainer.model.state_dict().items()}
            # Create a new instance to store the best state_dict to avoid modifying the trainer's model directly
            if best_model is None: # Initialize best_model structure if it's the first time
                best_model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=8,
                    ignore_mismatched_sizes=True
                )
                best_model.resize_token_embeddings(len(tokenizer))
            best_model.load_state_dict(best_model_state_dict)
            print(f"Updated best model with F1-micro: {best_f1:.4f}") #


        # Prediction
        print("\nGenerating predictions for test set...") #
        fold_preds = trainer.predict(test_dataset).predictions.argmax(-1) #
        all_preds.append(fold_preds) #
        print(f"Fold {fold+1} Predictions Sample:", fold_preds[:5]) #
        print(f"Class Distribution:", np.bincount(fold_preds)) #

        # Clean up
        del model, trainer #
        if torch.cuda.is_available():
            torch.cuda.empty_cache() #

    # Ensemble predictions
    print("\nEnsembling predictions from all folds...") #
    all_preds_array = np.array(all_preds) #

    # Calculate mode ACROSS FOLDS (axis=0)
    final_preds, _ = mode(all_preds_array, axis=0) #
    final_preds = final_preds.flatten().astype(int) #

    print(f"Final Predictions Sample:", final_preds[:5]) #
    print(f"Final Class Distribution:", np.bincount(final_preds)) #

    # The best_model stored should already have contiguous parameters
    # because we ensured contiguousness before loading the state_dict
    # No need to call make_model_contiguous again here if the above logic is correct.

    return best_model, tokenizer, final_preds

def main():
    """
    Main function to run the training process
    """
    print("Starting MathBERT training process...")
    
    # Load data
    train_df, test_df = load_data("../code/data/train.csv", "../code/data/test.csv")
    
    # Train model
    best_model, tokenizer, final_preds = train_mathbert(train_df, test_df)
    
    # Save the best model
    print(f"\nSaving the best model to {OUTPUT_DIR}...")
    save_model(best_model, tokenizer, OUTPUT_DIR)
    
    # Create submission
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'label': final_preds
    })
    
    # Save submission
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    print("\nTraining process completed successfully!")

if __name__ == "__main__":
    main()
