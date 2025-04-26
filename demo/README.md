# Math Question Classifier Demo

This demo showcases a model that classifies mathematical questions into 8 different categories using a fine-tuned MathBERT model.

## Categories
- **0**: Algebra
- **1**: Geometry
- **2**: Number Theory
- **3**: Combinatorics
- **4**: Calculus
- **5**: Probability & Statistics
- **6**: Linear Algebra
- **7**: Discrete Mathematics

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo:
   ```bash
   streamlit run app.py
   ```

## Files in this Directory

- `app.py`: Streamlit application for the demo interface
- `model_utils.py`: Utility functions for loading the model and making predictions
- `train.py`: Script for training the model with cross-validation
- `augment.py`: Data augmentation utilities for improving model performance
- `requirements.txt`: List of dependencies needed to run the demo

## Using the Demo

1. Enter a mathematical question in the text area
2. Click "Classify Question" to see the prediction
3. The prediction results will display:
   - The predicted category
   - A confidence distribution chart
   - A table with probabilities for each category

## Sample Questions

The demo includes sample questions for each category that you can try with one click.

## Model Details

- Base Model: MathBERT
- Fine-tuned with cross-validation
- F1-score micro average used for evaluation