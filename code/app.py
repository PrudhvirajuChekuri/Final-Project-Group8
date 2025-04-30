import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import sys
import os
import time
# Import necessary components from the refactored model_utils
from model_utils import preprocess_text, predict, ModelManager, model_dropdown

# prevent Streamlit from trying to load custom C++ classes (Keep this workaround)
try:
    torch.classes.__path__ = []
except Exception:
    # Handle cases where torch.classes might not be available or writable
    print("Warning: Could not modify torch.classes.__path__.")
    pass


# Fix for asyncio issue in Python 3.12 (Keep this)
if sys.version_info[0] == 3 and sys.version_info[1] >= 12:
    if sys.platform == "win32": # Policy only needed on Windows
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception as e:
            print(f"Could not set WindowsSelectorEventLoopPolicy: {e}")
            pass

# Add the parent directory ('code') to the path if needed, assuming app.py is in a subdir
# This allows importing model_utils if it's in the parent dir relative to app.py
# Adjust if your structure is different
# Example: Assuming app.py is in 'src' and model_utils.py is in 'src' as well,
# or if model_utils.py is in the root and app.py is in 'src', this might be needed.
# If both are in the same directory, it might not be necessary.
# sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Uncomment/adjust if needed

# --- Configuration ---
# Set page configuration
st.set_page_config(
    page_title="Math Question Classifier",
    page_icon="🧮", # Updated icon
    layout="wide"
)

# Define the fixed list of model names (must match subdirectory names in /output/models)
# Ensure these directories actually exist and contain valid model/tokenizer files
FIXED_MODEL_NAMES = ['ensemble/llama_1b_model', 'ensemble/deberta-model', 'ensemble/t5-model', 'ensemble/steroids'] 

# Define the base directory where model subdirectories are located
# Relative path from this app.py file to the /output/models directory
# Adjust '..' if app.py is nested differently relative to 'output/models'
# Example: If app.py is in root, and models are in output/models:
# MODEL_BASE_DIR = "output/models"
# Example: If app.py is in src/, and models are in output/models:
MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), "../output/models/")


# Define categories (assuming they are the same for all models)
CATEGORIES = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

# --- Model Initialization ---
# Use session state to store and manage the ModelManager instance
# This prevents re-initialization on every interaction/rerun
if 'model_manager' not in st.session_state:
    try:
        # Initialize ModelManager with the fixed list and base directory
        st.session_state.model_manager = ModelManager(
            fixed_model_names=FIXED_MODEL_NAMES,
            model_base_dir=MODEL_BASE_DIR,
            # Optionally set a default model name, otherwise first in list is used
            default_model_name=FIXED_MODEL_NAMES[0] # Load the first model by default
        )
        print("ModelManager initialized and stored in session state.")
    except Exception as e:
        # Display a fatal error if the manager can't even be initialized
        st.error(f"Fatal Error: Could not initialize Model Manager. Please check model paths and configurations. Details: {e}")
        # Optionally log the full traceback here
        st.stop() # Stop the app if manager fails to initialize

# Get the manager instance from session state
model_manager = st.session_state.model_manager

# Retrieve current model components (might be None if initial load failed or no model selected yet)
try:
    # Get the current state from the manager
    model, tokenizer, current_model_name, model_path = model_manager.get_current()
except Exception as e:
    # Handle potential errors during state retrieval, though less likely now
    st.error(f"Error retrieving current model state from manager: {e}")
    model, tokenizer, current_model_name, model_path = None, None, None, None


# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses transformer models to classify mathematical questions into 8 categories.
    Select a model from the dropdown in the main panel. Ensure the corresponding model files exist locally.
    """)
    
    st.header("Model Information")
    # Display info about the currently loaded model
    if model and tokenizer and current_model_name:
        st.success(f"✅ Model Loaded: **{current_model_name}**")
        # Show the path from where the model was loaded
        st.caption(f"Path: `{model_path}`")
        # Display model type and architecture if available in config
        st.write("Model type:", type(model).__name__)
        if hasattr(model.config, 'architectures') and model.config.architectures:
            st.write("Architecture:", model.config.architectures[0])
        else:
             st.write("Architecture: Not specified in config")
        # Show the device the model is loaded on
        try:
            device = next(model.parameters()).device
            st.write("Device:", device)
        except Exception:
             st.write("Device: Could not determine")
    elif model_manager.current_model_name: # If a name is set but model object is None (load failed)
         st.warning(f"⚠️ Attempted to load: **{model_manager.current_model_name}** but failed. Check logs and model files. Please select another model.")
    else: # If no model has been successfully loaded yet
        st.warning("⚠️ No model is currently loaded. Please select one from the dropdown.")

    st.header("Categories")
    # Display the defined categories
    for cat_id, cat_name in CATEGORIES.items():
        st.write(f"**{cat_id}**: {cat_name}")



# --- Main Page Content ---
st.title("Mathematical Question Classifier")
st.markdown("""
Enter a mathematical question below or choose an example, then click 'Classify Question'.
You can switch between available classification models using the dropdown.
""")

# --- Model Selection Dropdown ---
# Use the model_dropdown function from model_utils
# This function handles the selectbox UI and triggers model loading via the manager
selected_model_name = model_dropdown(model_manager, key="main_model_selector")

# Refresh model state after dropdown interaction, as it might trigger a reload via st.rerun()
model, tokenizer, current_model_name, model_path = model_manager.get_current()

# --- Question Input ---
st.header("Enter Question")

# Initialize session state for the question if it doesn't exist
if 'question' not in st.session_state:
    st.session_state.question = ""

# Text input area - uses session state for persistence across reruns
question_input = st.text_area(
    "Mathematical question:",
    height=150,
    placeholder="Example: What is the integral of 1/x dx?",
    # Use the value from session state
    value=st.session_state.question,
    key="question_input_area" # Unique key for this widget
)

# Update session state if text area content changes by the user
# This ensures the classify button uses the most recent input
if question_input != st.session_state.question:
    st.session_state.question = question_input
    # We don't rerun here, wait for the button click or sample selection

# --- Classify Button ---
# Use columns for layout control (centering the button)
col1_btn, col2_btn, col3_btn = st.columns([2, 6, 2]) # Adjust ratio as needed for centering

with col2_btn: # Place button in the middle column
    classify_clicked = st.button(
        "Classify Question",
        key="classify_btn",
        use_container_width=True, # Make button fill the column width
        type="primary", # Use Streamlit's primary button styling
        # Disable the button if no model/tokenizer is currently loaded
        disabled=(model is None or tokenizer is None)
    )
    # Provide feedback if the button is disabled
    if model is None or tokenizer is None:
        st.warning("Please select a valid model from the dropdown before classifying.")

# --- Button Styling (Keep as is for visual appeal) ---
st.markdown("""
<style>
    /* Target the button within the Streamlit button container */
    div[data-testid="stButton"] > button {
        font-size: 1.2rem;          /* Larger font size */
        font-weight: bold;          /* Make text bold */
        padding: 0.8rem 1.5rem;     /* Increase padding */
        border-radius: 10px;        /* Rounded corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        transition: all 0.3s ease; /* Smooth transition for hover */
    }
    /* Style for hover effect, only when the button is enabled */
    div[data-testid="stButton"] > button:hover:enabled {
        transform: translateY(-2px); /* Slight lift on hover */
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); /* Larger shadow on hover */
    }
</style>
""", unsafe_allow_html=True)


# --- Sample Questions ---
st.markdown("---") # Visual separator
st.markdown("### Or try one of these examples:")

# List of sample questions covering different categories
sample_questions = [
    "Solve the equation: 3x + 5 = 20", # Algebra
    "Find the area of a circle with radius 5 units", # Geometry
    "Prove that there are infinitely many prime numbers", # Number Theory
    "How many ways can 5 people be arranged in a line?", # Combinatorics
    "Calculate the limit of (sin x)/x as x approaches 0", # Calculus
    "If a fair coin is flipped 10 times, what is the probability of getting exactly 7 heads?", # Probability
    "Find the eigenvalues of the matrix [[1, 2], [3, 4]]", # Linear Algebra
    "Is the graph K5 planar?", # Discrete Mathematics (Graph Theory)
]

# Create 4 columns for the sample buttons for better layout
cols_samples = st.columns(4)
num_samples = len(sample_questions)

# Distribute the sample buttons across the columns
for i in range(num_samples):
    col_index = i % 4 # Cycle through columns 0, 1, 2, 3
    # Create a button for each sample question
    if cols_samples[col_index].button(f"Sample {i+1}", key=f"sample{i+1}"):
        # If a sample button is clicked, update the session state
        st.session_state.question = sample_questions[i]
        # Rerun the script immediately to update the text area display
        st.rerun()


# --- Prediction Logic ---
# This block executes only if the 'Classify Question' button was clicked
# and if a model and tokenizer are successfully loaded
if classify_clicked and (model is not None and tokenizer is not None):
    # Get the current question from session state (updated by text area or sample buttons)
    current_question = st.session_state.get("question", "").strip()

    if current_question: # Proceed only if there is a question entered
        # Show a spinner while processing
        with st.spinner(f"Processing with model '{current_model_name}'..."):
            try:
                start_time = time.time() # Start total timer

                # 1. Preprocess the question text
                preprocessing_start = time.time()
                processed_text = preprocess_text(current_question)
                preprocessing_time = time.time() - preprocessing_start

                # 2. Make prediction using the current model and tokenizer
                inference_start = time.time()
                if model_manager.current_model_name == 'ensemble/steroids':
                    pred_label_llama = predict(processed_text, model_name='ensemble/llama_1b_model', model=model_manager.models_ensemble[0], tokenizer=model_manager.tokenizers_ensemble[0])
                    pred_label_deberta = predict(processed_text, model_name='ensemble/deberta-model', model=model_manager.models_ensemble[1], tokenizer=model_manager.tokenizers_ensemble[1])
                    pred_label_t5 = predict(processed_text, model_name='ensemble/t5-model', model=model_manager.models_ensemble[2], tokenizer=model_manager.tokenizers_ensemble[2])
                    # hard voting 
                    pred_label = pred_label_llama if (pred_label_llama == pred_label_deberta or pred_label_llama == pred_label_t5) else (pred_label_deberta if pred_label_deberta == pred_label_t5 else pred_label_deberta)
                else:
                    pred_label = predict(processed_text, model_name=current_model_name, model=model, tokenizer=tokenizer)
                inference_time = time.time() - inference_start

                total_time = time.time() - start_time # End total timer

                # --- Display Results ---
                st.success(f"Prediction complete! (Total time: {total_time:.3f}s)")
                st.markdown("---") # Separator

                st.header("Results")
                # Get the category name using the predicted index
                predicted_category_name = CATEGORIES.get(pred_label, f"Unknown Category ({pred_label})")
                st.subheader(f"Predicted Category: **{predicted_category_name}**")

                # Display processing time metrics in columns
                st.subheader("Processing Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Total Time", f"{total_time:.3f} s")
                m_col2.metric("Preprocess Time", f"{preprocessing_time:.3f} s")
                m_col3.metric("Inference Time", f"{inference_time:.3f} s")
                st.markdown("---") # Separator

                # # Display probability distribution chart and table
                # st.subheader("Confidence Distribution")

                # # Create a Pandas DataFrame for easier plotting and display
                # probs_df = pd.DataFrame({
                #     # Map indices to category names
                #     'Category': [CATEGORIES.get(i, f"Unknown_{i}") for i in range(le))],
                #     'Probability'
                # })

                # # Sort the DataFrame by probability in descending order
                # probs_df = probs_df.sort_values('Probability', ascending=False).reset_index(drop=True)

                # # Displa as a bar chart using Streamlit's native chart
                # # Set 'Category' as the index for proper labeling
                # st.bar_chart(probs_df.set_index('Category')['Probability'])

                # # Display th in a table for precise values
                # st.write Table:")
                # # Format the 'Probability' column for better readability
                # st.dataframe(probs_df.style.format({"Probability": "{:.4f}"}))

            except ValueError as ve:
                 # Catch specific error from predict if model/tokenizer somehow became None after check
                 st.error(f"Prediction Error: {ve}. Model or tokenizer might have become unavailable.")
            except Exception as e:
                 # Catch any other errors during preprocessing or prediction
                 st.error(f"An error occurred during classification: {str(e)}")
                 # For debugging, you might want to print the traceback
                 # import traceback
                 # st.exception(e)

    else: # If classify button clicked but the question text area is empty
        st.warning("Please enter a question in the text area above.")

# --- Footer ---
st.markdown("---") # Final separator
st.markdown("Mathematical Question Classifier Demo")
