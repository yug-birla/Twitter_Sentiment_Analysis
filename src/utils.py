"""
Utility functions for loading and inspecting data
"""

import pandas as pd
from src import config


def load_dataset():
    """
    Load the Twitter sentiment dataset and prepare basic structure
    """
    try:
        # Read CSV file
        # encoding='latin-1' is required for this dataset
        # header=None because dataset doesn't contain column names
        df = pd.read_csv(config.DATASET_PATH, encoding='latin-1', header=None)

        # Assign proper column names based on dataset format
        df.columns = ["target", "id", "date", "flag", "user", "text"]

        # Keep only relevant columns for sentiment analysis
        df = df[["target", "text"]]

        # Convert labels:
        # 4 → positive (1)
        # 0 → negative (0)
        df["label"] = df["target"].apply(lambda x: 1 if x == 4 else 0)

        # Optional: take a sample to speed up development
        df = df.sample(100000, random_state=config.RANDOM_STATE)

        # Print confirmation and preview
        print("Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())

        return df

    except FileNotFoundError:
        print(f"Dataset not found at: {config.DATASET_PATH}")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def get_data_info(df):
    """
    Display basic dataset statistics
    """
    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    # Shape (rows, columns)
    print(f"\nShape: {df.shape}")

    # Data types of each column
    print(f"\nData types:\n{df.dtypes}")

    # Check missing/null values
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Distribution of original labels
    print(f"\nOriginal target distribution:\n{df['target'].value_counts()}")

    # Distribution of processed labels (0/1)
    print(f"\nBinary label distribution:\n{df['label'].value_counts()}")