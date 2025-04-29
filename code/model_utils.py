import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
import gdown

# Constants
MODEL_NAME = "tbs17/MathBERT"
MAX_LENGTH = 256
MODEL_PATH = "../code/output"  # Path to the saved model

def load_model():
    """
    Load the trained MathBERT model and tokenizer
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Initialize tokenizer with math special tokens
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_special_tokens({'additional_special_tokens': ['[MATH]']})
        flag = False
        model_path = ""
        
        # If a safetensors file isn’t already downloaded, grab it from Google Drive

        # Replace with your actual Google Drive file ID for the safetensors archive
        GDRIVE_FILE_ID = "1l9KfJ45C90QMsIRy0mwH_L1iUTiqoGjv"

        # Path where the safetensors checkpoint should live
        local_model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), MODEL_PATH, "model")
        )
        safetensors_path = os.path.join(local_model_dir, "model.safetensors")

        if not os.path.exists(safetensors_path):
            os.makedirs(local_model_dir, exist_ok=True)
            download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            print(f"Downloading safetensors checkpoint from Google Drive …")
            gdown.download(download_url, safetensors_path, quiet=False)
            print("Download complete!")
            
        # Use absolute path to be sure we find the model
        abs_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
        model_dir = os.path.join(abs_model_path, "model")
        
        # Load the model from the saved path if available
        if os.path.exists(model_dir):
            print(f"Loading fine-tuned model from {model_dir}...")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=8
                )
                model_path = model_dir
                print("Model loaded successfully!")
                flag = True
            except Exception as err:
                print(f"Failed loading fine-tuned model ({err}); falling back to base model.")
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=8,
                    ignore_mismatched_sizes=True
                )
                model_path = MODEL_NAME
                model.resize_token_embeddings(len(tokenizer))
        else:
            # Fall back to the base model if the fine-tuned model is not available
            print(f"Fine-tuned model not found at {model_dir}! \n\nLoading base MathBERT model instead...")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=8,
                ignore_mismatched_sizes=True
            )
            model_path = MODEL_NAME
            model.resize_token_embeddings(len(tokenizer))
            
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set model to evaluation mode
        
        return model, tokenizer, flag, model_path
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_text(text):
    """
    Preprocess the input text to match the format used during training
    
    Args:
        text (str): Input mathematical question
        
    Returns:
        str: Preprocessed text
    """
    # Preserve mathematical notation
    text = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', text)
    text = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', text)
    return text.strip()

def predict(text, model, tokenizer):
    """
    Make a prediction for the given text
    
    Args:
        text (str): Preprocessed input text
        model: The loaded model
        tokenizer: The loaded tokenizer
        
    Returns:
        tuple: (predicted_class, class_probabilities)
    """
    try:
        # Tokenize the input
        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Get probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get the predicted class
        predicted_class = np.argmax(probabilities)
        
        return int(predicted_class), probabilities.tolist()
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


def save_model(model: PreTrainedModel, tokenizer, output_dir):
    """
    Save the model and tokenizer
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: The directory to save to
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # If wrapped in DataParallel
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.to("cpu")

    # Ensure every parameter is contiguous
    for name, param in model_to_save.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
    # …and every buffer (e.g. LayerNorm running stats) too
    for name, buf in model_to_save.named_buffers():
        if not buf.data.is_contiguous():
            buf.data = buf.data.contiguous()

    # Now this will pass the Safetensors check
    model_to_save.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(output_dir)
