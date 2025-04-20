import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import sys
import os

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

# Add the src directory to the path so we can import from it
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

if col1.button("Sample 1", key="sample1"):
    question = sample_questions[0]
    st.session_state.question = question
    st.rerun() # Add this to force a re-render
if col1.button("Sample 2", key="sample2"):
    question = sample_questions[1]
    st.session_state.question = question
    st.rerun()
if col1.button("Sample 3", key="sample3"):
    question = sample_questions[2]
    st.session_state.question = question
    st.rerun()
if col1.button("Sample 4", key="sample4"):
    question = sample_questions[3]
    st.session_state.question = question
    st.rerun()
if col2.button("Sample 5", key="sample5"):
    question = sample_questions[4]
    st.session_state.question = question
    st.rerun()
if col2.button("Sample 6", key="sample6"):
    question = sample_questions[5]
    st.session_state.question = question
    st.rerun()
if col2.button("Sample 7", key="sample7"):
    question = sample_questions[6]
    st.session_state.question = question
    st.rerun()
if col2.button("Sample 8", key="sample8"):
    question = sample_questions[7]
    st.session_state.question = question
    st.rerun()

# Check if we have a question from session state
if 'question' in st.session_state:
    question = st.session_state.question
    # Display the question in the text area
    st.query_params["question"]=question

# Process button
# Process button - replace the entire button section with this
classify_clicked = st.button("Classify Question", key="classify_btn")
if classify_clicked:
    if st.session_state.question:
        with st.spinner("Processing..."):
            try:
                # Load model if not already loaded
                model, tokenizer, flag, model_path = get_model()
                
                # Preprocess the question
                processed_text = preprocess_text(st.session_state.question)
                
                # Make prediction
                result = predict(processed_text, model, tokenizer)
                print(f"Result type: {type(result)}, contains: {result}")
                # Then unpack
                prediction, probabilities = result
                
                # Display results
                st.success(f"Prediction complete!")
                
                # Show the predicted category
                st.header("Results")
                st.subheader(f"Predicted Category: {categories[prediction]}")
                
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
