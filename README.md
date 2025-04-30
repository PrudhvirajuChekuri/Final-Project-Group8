# Math Problem Classification Project

## Overview

This project explores various approaches to classify math problems into one of eight categories based on the dataset from the Kaggle competition: [Classification of Math Problems by Kasut Academy](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/overview). It includes data augmentation using AWS Bedrock and fine-tuning of different transformer models, including sequence classifiers, sequence-to-sequence models, and instruction-following LLMs. An ensemble approach is also considered.

## Prerequisites

* Python environment with standard data science libraries (pandas, numpy, etc.).
* PyTorch.
* Hugging Face Libraries: transformers, datasets, evaluate, trl.
* unsloth: For efficient Llama model fine-tuning.
* boto3: For data generation scripts using AWS Bedrock. Requires configured AWS credentials with Bedrock access.
* Original competition data files: train.csv, test.csv.

## Execution Order and Script Descriptions

The recommended order for running the scripts is as follows:

1.  Data Acquisition (Manual)
    * Download train.csv and test.csv from the Kaggle competition page linked above. Place them in the root directory or update paths within the notebooks accordingly.

2.  Data Augmentation (Optional)
    * Run one of the following notebooks if you wish to generate augmented data. This is required for the LLAMA_1B.ipynb notebook. Note: These require configured AWS credentials for Bedrock access.
        * *data_generation_generic.ipynb: Generates synthetic math problems using AWS Bedrock (Claude 3.5 Sonnet) with a generic prompt. Aims to balance the class distribution by creating new samples for underrepresented categories based on examples from train.csv. Saves output to augmented_train_.csv.
        * *data_generation_specific.ipynb: Also uses AWS Bedrock (Claude 3.5 Sonnet) to generate synthetic data based on a subset of train.csv (train.csv[:9189]). The prompt specifically requests that the generated problems mimic the use of LaTeX/math symbols found in the example problems provided, aiming for higher stylistic fidelity. Saves output to augmented_train_.csv.
    * Note: The Llama notebook uses a file named augmented_train_cleaned.csv. You may need to rename the output of the generation scripts or perform additional cleaning steps.

3.  Model Training and Evaluation
    * These notebooks fine-tune different models. They can generally be run independently, provided the required data (original train.csv, and augmented data for Llama) is available. Each notebook typically includes preprocessing, training, evaluation on a validation/test split, and generation of a submission.csv file for the competition's test set.
        * *nlp-final-project-nn-distillbert-base.ipynb*: Fine-tunes distilbert-base-uncased (an encoder-only model) for sequence classification using the original train.csv.
        * *nlp-final-project-nn-deberta-v3.ipynb*: Fine-tunes microsoft/deberta-v3-base (an encoder-only model) for sequence classification using the original train.csv. Includes generation of submission.csv.
        * *nlp-final-project-t5.ipynb*: Fine-tunes t5-base (an encoder-decoder model) using a sequence-to-sequence approach where the model generates the textual label name (e.g., "Algebra"). Uses the original train.csv. Includes generation of submission.csv.
        * *LLAMA_1B.ipynb: Fine-tunes unsloth/Llama-3.2-1B (a decoder-only model) using the Unsloth library and LoRA. It treats the task as instruction-following, generating the textual label name. **Requires both train.csv and the augmented data CSV* (e.g., augmented_train_cleaned.csv). Includes evaluation on a hold-out set and generation of submission.csv.

4.  Ensemble Model (Conceptual)
    * Combine the predictions from multiple trained models (e.g., DeBERTa, T5, Llama) using a hard voting strategy.
    * Note: A specific script for ensembling is not provided. This would require custom code to load predictions or saved models from the previous steps and implement the voting logic.
