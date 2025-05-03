import pandas as pd
import re
import random
import argparse
import os
from textattack.augmentation import Augmenter
from textattack.transformations import (
    WordSwapRandomCharacterDeletion,
    WordSwapChangeLocation
)
from textattack.transformations import CompositeTransformation

class MathAugmenter:
    """
    Class for augmenting mathematical text data while preserving mathematical structure
    """
    def __init__(self, num_augments=2):
        """
        Initialize the augmenter
        
        Args:
            num_augments (int): Number of augmented versions to generate per sample
        """
        self.num_augments = num_augments
        
        # Create a composite transformation
        transformation = CompositeTransformation([
            WordSwapRandomCharacterDeletion(random_one=True),
            WordSwapChangeLocation()
        ])
        
        # Initialize the augmenter
        self.augmenter = Augmenter(
            transformation=transformation,
            transformations_per_example=1  # Generate 1 augmented version per call
        )

    def augment_math_problem(self, text):
        """
        Augment while preserving mathematical structure
        
        Args:
            text (str): The mathematical text to augment
            
        Returns:
            list: List of augmented texts
        """
        try:
            # Find all equations enclosed in $ signs
            equations = re.findall(r'\$(.*?)\$', text, re.DOTALL)
            placeholders = [f' EQUATION_{i} ' for i in range(len(equations))]
            
            # Create template with placeholders
            template = re.sub(r'\$(.*?)\$', lambda m: placeholders.pop(0), text)
            
            # Get list of augmented texts
            augmented_texts = self.augmenter.augment(template)
            
            # Restore equations in all augmented versions
            processed = []
            for aug_text in augmented_texts:
                for i, eq in enumerate(equations):
                    aug_text = aug_text.replace(f'EQUATION_{i}', f'${eq}$')
                processed.append(aug_text)
                
            return processed
            
        except Exception as e:
            print(f"Augmentation failed for text: {text[:50]}... | Error: {e}")
            return [text]  # Return original as fallback

def augment_data(input_file, output_file, minority_classes=None, augment_factor=3):
    """
    Augment the data and save to a new file
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the augmented CSV file
        minority_classes (list): List of class IDs to augment (default: [6, 7])
        augment_factor (int): Number of augmented versions to generate per sample
    """
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    print("Original class distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Default minority classes if not specified
    if minority_classes is None:
        minority_classes = [6, 7]
    
    print(f"\nAugmenting minority classes: {minority_classes}")
    print(f"Augmentation factor: {augment_factor}")
    
    # Initialize augmenter
    augmenter = MathAugmenter()
    
    # Augment each minority class
    for class_id in minority_classes:
        print(f"\nProcessing class {class_id}...")
        class_samples = df[df['label'] == class_id]['Question'].tolist()
        print(f"Found {len(class_samples)} samples")
        
        augmented_samples = []
        
        for i, sample in enumerate(class_samples):
            # Get multiple augmented versions
            augmented_versions = augmenter.augment_math_problem(sample)
            augmented_samples.extend(augmented_versions[:augment_factor])
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(class_samples) - 1:
                print(f"Processed {i + 1}/{len(class_samples)} samples")
        
        print(f"Generated {len(augmented_samples)} augmented samples")
        
        # Add to training data
        new_rows = pd.DataFrame({
            'Question': augmented_samples,
            'label': [class_id] * len(augmented_samples)
        })
        df = pd.concat([df, new_rows], ignore_index=True)
    
    print("\nNew class distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Save augmented data
    print(f"\nSaving augmented data to {output_file}")
    df.to_csv(output_file, index=False)
    print("Done!")

def main():
    """
    Main function to run the data augmentation process
    """
    parser = argparse.ArgumentParser(description='Augment mathematical text data')
    parser.add_argument('--input', type=str, default='../src/data/train.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='../src/data/train_augmented.csv',
                        help='Path to save the augmented CSV file')
    parser.add_argument('--classes', type=int, nargs='+', default=[6, 7],
                        help='List of class IDs to augment')
    parser.add_argument('--factor', type=int, default=3,
                        help='Number of augmented versions to generate per sample')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run augmentation
    augment_data(args.input, args.output, args.classes, args.factor)

if __name__ == "__main__":
    main()
