import os
import re
import torch
import numpy as np
# import gdown # Removed: No longer downloading from GDrive
import streamlit as st
import gc
import time
import random
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel
)
from unsloth import FastLanguageModel

# Constants
# BASE_MODEL_NAME = "tbs17/MathBERT" # Keep if needed as a potential fallback or reference
MAX_LENGTH = 256
# Default directory where ModelManager looks for model subdirectories
DEFAULT_MODEL_BASE_DIR = "/home/ubuntu/github_NLP/code/output/models" # Changed path

id2label = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}
label2id = {v: k for k, v in id2label.items()}

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

# Removed Google Drive ID
# GDRIVE_FILE_ID = "1l9KfJ45C90QMsIRy0mwH_L1iUTiqoGjv"

def set_random_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
    # Add determinism settings if needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_random_seed()

# Removed global Llama model loading block

class ModelManager:
    """
    Manages loading and switching between a fixed list of locally stored models.
    Assumes models are for sequence classification and stored in subdirectories
    under the specified model_base_dir.
    """
    def __init__(self, fixed_model_names: list[str], model_base_dir=DEFAULT_MODEL_BASE_DIR, default_model_name='ensemble/llama_1b_model'):
        """
        Initializes the ModelManager with a fixed list of models.

        Args:
            fixed_model_names (list[str]): A list of model names. Each name must
                                           correspond to a subdirectory within model_base_dir
                                           containing a saved Hugging Face model and tokenizer.
            model_base_dir (str): The base directory containing the model subdirectories.
            default_model_name (str, optional): The name of the model to load initially.
                                                If None, the first model in the list is used.
        """
        if not fixed_model_names:
            raise ValueError("fixed_model_names list cannot be empty.")

        # Use absolute path relative to this file's location for robustness
        script_dir = os.path.dirname(__file__)
        self.model_base_dir = os.path.abspath(os.path.join(script_dir, model_base_dir))
        print(f"Model base directory set to: {self.model_base_dir}")
        # Note: We don't create the base dir here, we assume it exists with model subdirs

        self.available_models = fixed_model_names # Use the provided fixed list
        self.model = None
        self.tokenizer = None
        self.current_model_name = 'mathBERT' # Default to MathBERT
        self.current_model_path = '/home/ubuntu/github_NLP/code/output/models/mathBERT' # Path of the currently loaded model
        self.models_ensemble = []
        self.tokenizers_ensemble = []

        # Determine the default model to load
        initial_model_to_load = default_model_name if default_model_name in self.available_models else self.available_models[0]

        # Attempt to load the initial model
        try:
            print(f"Attempting to load initial model: {initial_model_to_load}")
            self.load_model(initial_model_to_load)
        except Exception as e:
            print(f"CRITICAL WARNING: Failed to load initial model '{initial_model_to_load}': {e}. No model loaded.")
            self._reset_state() # Ensure clean state if initial load fails

    def _reset_state(self):
        """Resets model-related state variables."""
        self.model = None
        self.tokenizer = None
        self.current_model_name = 'mathBERT' # Reset to default
        self.current_model_path = None
        self.models_ensemble = []
        self.tokenizers_ensemble = []
        print("Model manager state reset.")


    def load_model(self, model_name):
        """
        Loads a model and its tokenizer specified by name from the fixed list.
        Assumes the model is stored locally in a subdirectory under model_base_dir.
        """
        if model_name not in self.available_models:
             raise ValueError(f"Model '{model_name}' is not in the predefined list of available models: {self.available_models}")

        if model_name == self.current_model_name and self.model is not None and self.tokenizer is not None:
            print(f"Model '{model_name}' is already loaded.")
            return

        self.unload_model() # Unload previous model first
        print(f"Attempting to load model: {model_name}")

        load_path = os.path.join(DEFAULT_MODEL_BASE_DIR, model_name)
        
        print(f"Expected model path: {load_path}")

        # if not os.path.isdir(load_path):
        #     raise FileNotFoundError(f"Model directory not found for '{model_name}' at expected path: {load_path}")

        print(f"Loading model and tokenizer from local directory: {load_path}...")

        try:
            if model_name == 'mathBERT':
                # Load the MathBERT model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/github_NLP/code/output/models/mathBERT')
                self.model = AutoModelForSequenceClassification.from_pretrained('/home/ubuntu/github_NLP/code/output/models/mathBERT')
                # Move model to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                self.model.eval()
                print(f"MathBERT model and tokenizer loaded successfully on {device}.")
            elif model_name == 'ensemble/llama_1b_model':
                max_seq_length = 2048
                dtype = torch.float16
                load_in_4bit = False

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "/home/ubuntu/github_NLP/code/output/models/ensemble/llama_1b_model",
                    max_seq_length = max_seq_length,
                    dtype = dtype,
                    load_in_4bit = load_in_4bit
                )
                # Update the current model state
                self.current_model_name = model_name
                self.current_model_path = load_path
                print(f"Successfully loaded Llama model '{model_name}'") # Add success message

            elif model_name == 'ensemble/t5-model':
                t5_model_dir = '/home/ubuntu/github_NLP/code/output/models/ensemble/t5-model'
                print(f"\nLoading fine-tuned T5 model and tokenizer from {t5_model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_dir)

                device = 0
                self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_dir).to(f"cuda:{device}")
                self.model.eval()

                print("Model and tokenizer reloaded successfully.")
                
            elif model_name == 'ensemble/deberta-model':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device} to load deberta model")
                deberta_model_dir = '/home/ubuntu/github_NLP/code/output/models/ensemble/deberta-model'
                self.tokenizer = AutoTokenizer.from_pretrained(deberta_model_dir)
                print("Tokenizer loaded.")

                self.model = AutoModelForSequenceClassification.from_pretrained(deberta_model_dir)
                print("Model loaded.")
                self.model.to(device)
                print(f"Model moved to {device}.")

                self.model.eval()
            elif model_name == 'ensemble/steroids':
                models_list = ['ensemble/llama_1b_model', 'ensemble/deberta-model', 'ensemble/t5-model']
                tokenizers_ensemble = []
                models_ensemble = []
                for model_key in models_list:
                    self.load_model(model_key)
                    self.models_ensemble.append(self.model)
                    self.tokenizers_ensemble.append(self.tokenizer)
    

                print("Successfully loaded models and their tokenizers for 'ensemble/steroids'.")
            
                
        except Exception as e:
            print(f"Error loading model '{model_name}' from {load_path}: {str(e)}")
            self._reset_state() # Ensure clean state on failure
            # Re-raise the exception to signal the failure upstream
            raise Exception(f"Failed to load model '{model_name}': {str(e)}")


    def unload_model(self):
        """Unloads the current model and tokenizer, clears GPU cache."""
        if self.model is not None or self.tokenizer is not None:
            print(f"Unloading model: {self.current_model_name}")
            del self.model
            del self.models_ensemble
            del self.tokenizers_ensemble
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                print("Clearing CUDA cache...")
                torch.cuda.empty_cache()
                gc.collect() # Force garbage collection
            self._reset_state() # Reset state variables
            print("Model unloaded.")
        else:
            print("No model currently loaded.")

    def get_current(self):
        """Returns the current model, tokenizer, model name, and model path."""
        # The 'flag' indicating special fine-tuned model is removed.
        return self.model, self.tokenizer, self.current_model_name, self.current_model_path

    def get_available_models(self):
        """Returns the fixed list of available model names."""
        return self.available_models

# --- Dropdown Function (uses the refactored manager) ---
def model_dropdown(model_manager: ModelManager, key="model_selector"):
    """Creates a Streamlit dropdown to select and load models via ModelManager."""
    models = model_manager.get_available_models()
    if not models:
        st.warning("No models available for selection (ModelManager list is empty).")
        return None

    current_model_name = model_manager.current_model_name
    default_index = 0
    if current_model_name in models:
        default_index = models.index(current_model_name)
    elif models:
         print(f"Warning: Current model '{current_model_name}' not in available list {models}. Defaulting dropdown to index 0 ({models[0]}).")
         # Optionally try to load the default if current isn't available?
         # Or rely on the initial load in ModelManager.__init__
    else:
         # This case should not happen if __init__ checks for empty list
         st.error("Error: Model list is unexpectedly empty.")
         return None

    selected = st.selectbox(
        "Select Model",
        models,
        index=default_index,
        key=key,
        # on_change callback can be complex with Streamlit state,
        # sticking to check-after-selection.
    )

    # Load model only if selection changes *and* is different from the currently loaded model
    if selected and selected != model_manager.current_model_name:
        with st.spinner(f"Loading model '{selected}'... This may take a moment."):
            try:
                start_load_time = time.time()
                model_manager.load_model(selected)
                load_time = time.time() - start_load_time
                model_manager.current_model_name = selected
                st.success(f"Switched to model: {selected} (loaded in {load_time:.2f}s)")
                # Rerun might be needed depending on how state is managed elsewhere
                st.rerun() # Rerun to ensure the rest of the app uses the new model state
            except Exception as e:
                st.error(f"Failed to load model '{selected}': {e}")
                # Attempt to reload the previous model? Or leave in unloaded state?
                # For now, it will be in an unloaded state. User needs to select a working model.
                st.warning("Model loading failed. Please select another model.")

    return selected # Return the name selected in the dropdown

# Removed load_special_finetuned_model function

def preprocess_text(text):
    """
    Preprocess the input text.
    (Assuming this preprocessing is suitable for all models in the list)

    Args:
        text (str): Input mathematical question

    Returns:
        str: Preprocessed text
    """
    # Preserve mathematical notation - adjust if models need different handling
    processed = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', str(text)) # Ensure text is string
    processed = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', processed)
    # Basic cleaning: replace multiple spaces with one, strip leading/trailing space
    processed = re.sub(r'\s+', ' ', processed).strip()
    return processed

def llama_predict(text, model, tokenizer):
    FastLanguageModel.for_inference(model)
    instruction = "Classify this math problem into one of these eight topics: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology."

    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
    inputs = tokenizer(
        [
            prompt.format(
                instruction, # instruction
                text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    raw_output = tokenizer.batch_decode(outputs)[0]
        
    def parse_output(output):
        re_match = re.search(r'### Response:\n(.*?)<\|end_of_text\|>', output, re.DOTALL)
        if re_match:
            response = re_match.group(1).strip()
            return response
        else:
            return ''

    final_output = label2id.get(parse_output(raw_output), 0)
    return final_output

def t5_predict(text, model, tokenizer):
    MAX_TARGET_LENGTH = 32
    classifier_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)
    prefix = "Classify this math problem: "
    prefixed_test_question = prefix + text 

    raw_predictions = classifier_pipeline([prefixed_test_question], max_length=MAX_TARGET_LENGTH, clean_up_tokenization_spaces=True)

    predicted_label_names = raw_predictions[0]['generated_text'].strip()
    
    return label2id.get(predicted_label_names, 0)

def deberta_predict(text, model, tokenizer):
    cleaned_question = clean_math_text_final(text)
    comp_test_df = pd.DataFrame({'cleaned_question': [cleaned_question]})
    training_args = TrainingArguments(
        output_dir="./",
        push_to_hub=False,
        per_device_eval_batch_size=32,
        report_to="none",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )


    predict_dataset = Dataset.from_pandas(comp_test_df[['cleaned_question']])
    print("Test data converted to Dataset format.")
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

    print("\n--- Making Predictions ---")
    predictions_output = trainer.predict(tokenized_predict_dataset)

    logits = predictions_output.predictions

    predicted_labels = np.argmax(logits, axis=-1)
    print("Predictions generated.")

    deberta_labels = [i for i in predicted_labels]
    print("Predicted labels:", deberta_labels)
    return deberta_labels[0]

def mathBERT_predict(text, model, tokenizer):
    """
    Make a prediction for the given text using the loaded MathBERT model.

    Args:
        text (str): Preprocessed input text
        model: The loaded MathBERT model
        tokenizer: The loaded tokenizer corresponding to the model

    Returns:
        int: Predicted class index
    """
    # Determine the device the model is on
    device = next(model.parameters()).device
    print(f"MathBERT model is on device: {device}")


    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)

    # Move tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Input tensors moved to device: {device}")


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_index = torch.argmax(logits, dim=1).item()
    print(f"MathBERT prediction: {predicted_class_index}")
    return predicted_class_index


def predict(text, model_name, model, tokenizer):
    """
    Make a prediction for the given text using the loaded sequence classification model.

    Args:
        text (str): Preprocessed input text
        model: The loaded sequence classification model
        tokenizer: The loaded tokenizer corresponding to the model

    Returns:
        tuple: (predicted_class_index, class_probabilities_list)

    Raises:
        ValueError: If model or tokenizer is None.
        Exception: For errors during tokenization or model inference.
    """
    if model is None or tokenizer is None:
        raise ValueError("Model or Tokenizer is not loaded. Cannot predict.")

    try:
        if model_name == 'mathBERT':
            # Use the mathBERT_predict function for MathBERT model
            prediction = mathBERT_predict(text, model, tokenizer)
        elif model_name == 'ensemble/llama_1b_model':
            # Use the llama_predict function for Llama model
            prediction = llama_predict(text, model, tokenizer)
        elif model_name == 'ensemble/t5-model':
            # Use the t5_predict function for T5 model
            prediction = t5_predict(text, model, tokenizer)
            return prediction 
        elif model_name == 'ensemble/deberta-model':
            prediction = deberta_predict(text, model, tokenizer)
        return prediction
        

    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction for text '{text[:50]}...': {e}")
        # Re-raise a more specific exception
        raise Exception(f"Error during prediction: {e}")


def save_model(model: PreTrainedModel, tokenizer, output_dir):
    """
    Save the model and tokenizer to a specified directory.

    Args:
        model: The Hugging Face model to save.
        tokenizer: The Hugging Face tokenizer to save.
        output_dir: The directory path to save the model and tokenizer to.
                    A subdirectory named 'model' will be created if saving
                    in Safetensors format by default with save_pretrained.
                    Let's save directly into output_dir for consistency with loading.
    """
    if not isinstance(model, PreTrainedModel):
        raise TypeError("Model must be an instance of transformers.PreTrainedModel")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model and tokenizer to: {output_dir}")

    # Get the model ready for saving (move to CPU, handle DataParallel)
    model_to_save = model.module if hasattr(model, "module") else model
    original_device = next(model_to_save.parameters()).device # Store original device
    model_to_save.to("cpu")

    # Ensure parameters and buffers are contiguous (important for Safetensors)
    # This might not be strictly necessary with recent HF versions but is good practice
    # for name, param in model_to_save.named_parameters():
    #     if not param.data.is_contiguous():
    #         param.data = param.data.contiguous()
    # for name, buf in model_to_save.named_buffers():
    #     if not buf.data.is_contiguous():
    #         buf.data = buf.data.contiguous()

    try:
        # Save the model's weights, config, etc.
        # save_pretrained saves config.json, model weights (pytorch_model.bin or model.safetensors), etc.
        model_to_save.save_pretrained(output_dir) # Saves directly into output_dir

        # Save the tokenizer's files (tokenizer.json, vocab.txt/merges.txt, special_tokens_map.json, etc.)
        tokenizer.save_pretrained(output_dir)

        print("Model and tokenizer saved successfully.")

    except Exception as e:
        print(f"Error saving model/tokenizer to {output_dir}: {e}")
        raise # Re-raise the exception

    finally:
        # Move model back to its original device if needed
        model_to_save.to(original_device)
