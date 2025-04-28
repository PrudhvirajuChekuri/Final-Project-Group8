import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import sys
import os
import time

# prevent Streamlit from trying to load custom C++ classes
# This is a workaround for an issue with Streamlit and PyTorch custom classes
# source : https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
torch.classes.__path__ = []

# Fix for asyncio issue in Python 3.12
if sys.version_info[0] == 3 and sys.version_info[1] >= 12:
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass

# Add the code directory to the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_utils import load_model, preprocess_text, predict

# Set page configuration
st.set_page_config(
    page_title="Math Question Classifier",
    page_icon="➗",
    layout="wide"
)

# Title and description
st.title("Mathematical Question Classifier")
st.markdown("""
This demo showcases a model that classifies mathematical questions into 8 different categories.
Enter a mathematical question below to see the model's prediction.
""")

# Load the model
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, flag, model_path = get_model()

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses a fine-tuned MathBERT model to classify mathematical questions into 8 categories.
    
    The model was trained on a dataset of mathematical problems using cross-validation.
    """)
    
    st.header("Categories")
    categories = {
        0: "Algebra",
        1: "Geometry",
        2: "Number Theory",
        3: "Combinatorics",
        4: "Calculus",
        5: "Probability & Statistics",
        6: "Linear Algebra",
        7: "Discrete Mathematics"
    }
    
    for cat_id, cat_name in categories.items():
        st.write(f"**{cat_id}**: {cat_name}")

    st.header("Model Information")
    if flag:
        st.success(f"✅ Using fine-tuned model from:\n{model_path}")
    else:
        st.warning(f"⚠️ Fine-tuned model not found!\n\nUsing base MathBERT model instead.")
    
    # Show model architecture info
    st.write("Model type:", type(model).__name__)
    if hasattr(model.config, 'architectures'):
        st.write("Architecture:", model.config.architectures[0])
    
    # Show device info
    device = next(model.parameters()).device
    st.write("Device:", device)


# Main content
st.header("Question Classification")

# First, check if we have a question in session state before creating the text area
if 'question' not in st.session_state:
    st.session_state.question = ""

# Text input for the question
question = st.text_area(
    "Enter a mathematical question:",
    height=150,
    placeholder="Example: Find the derivative of f(x) = x^2 + 3x + 2",
    value=st.session_state.question,
    key="question_input"  # Add a key to track this input
)

# Update session state whenever text is entered
if question:
    st.session_state.question = question

# Using columns to control button width (60%)
col1, col2, col3 = st.columns([2, 6, 2])  # 2:6:2 ratio gives us 60% in middle column

with col2:  # Place button in middle column
    # Process button with custom style
    classify_clicked = st.button(
        "Classify Question", 
        key="classify_btn",
        use_container_width=True,  # Fill the column (which is 60% of total width)
        type="primary"  # Use primary button style (blue color)
    )

# Add custom CSS to make the button more prominent
st.markdown("""
<style>
    div[data-testid="stButton"] button {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# Sample questions
st.markdown("### Or try one of these examples:")
col1, col2 = st.columns(2)

sample_questions = [
    "Solve the equation: 3x + 5 = 20",
    "Find the area of a circle with radius 5 units",
    "Prove that there are infinitely many prime numbers",
    "How many ways can 5 people be arranged in a line?",
    "Calculate the limit of (sin x)/x as x approaches 0",
    "If a fair coin is flipped 10 times, what is the probability of getting exactly 7 heads?",
    "Find the eigenvalues of the matrix [[1, 2], [3, 4]]",
    "Determine if the graph with vertices {1,2,3,4,5} and edges {(1,2), (2,3), (3,4), (4,5), (5,1)} is bipartite"
]

col1, col2, col3, col4 = st.columns(4)

# Define columns for each sample question (4 columns, 2 rows)
columns = [col1, col2, col3, col4, col1, col2, col3, col4]

# Use a loop to create buttons for all 8 samples
for i in range(8):
    if columns[i].button(f"Sample {i+1}", key=f"sample{i+1}"):
        st.session_state.question = sample_questions[i]
        st.rerun()

# Check if we have a question from session state
if 'question' in st.session_state:
    question = st.session_state.question
    # Display the question in the text area
    st.query_params["question"]=question

if classify_clicked:
    if st.session_state.question:
        with st.spinner("Processing..."):
            try:
                # Start timing
                start_time = time.time()
                
                # Load model if not already loaded
                model, tokenizer, flag, model_path = get_model()
                
                # Preprocess the question
                preprocessing_start = time.time()
                processed_text = preprocess_text(st.session_state.question)
                preprocessing_time = time.time() - preprocessing_start
                
                # Make prediction
                inference_start = time.time()
                result = predict(processed_text, model, tokenizer)
                inference_time = time.time() - inference_start
                
                # Calculate total processing time
                total_time = time.time() - start_time
                
                print(f"Result type: {type(result)}, contains: {result}")
                # Then unpack
                prediction, probabilities = result
                
                # Display results
                st.success(f"Prediction complete!")
                
                # Show the predicted category
                st.header("Results")
                st.subheader(f"Predicted Category: {categories[prediction]}")
                
                # Display processing time metrics
                st.subheader("Processing Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Processing Time", f"{total_time:.3f} sec")
                col2.metric("Preprocessing Time", f"{preprocessing_time:.3f} sec")
                col3.metric("Inference Time", f"{inference_time:.3f} sec")
                
                # Display probability distribution
                st.subheader("Confidence Distribution")
                
                # Create a dataframe for the probabilities
                probs_df = pd.DataFrame({
                    'Category': [categories[i] for i in range(8)],
                    'Probability': probabilities
                })
                
                # Sort by probability
                probs_df = probs_df.sort_values('Probability', ascending=False)
                
                # Display as a bar chart
                st.bar_chart(probs_df.set_index('Category'))
                
                # Display as a table
                st.table(probs_df)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question first.")

# Footer
st.markdown("---")
st.markdown("Mathematical Question Classifier Demo | Powered by MathBERT")
