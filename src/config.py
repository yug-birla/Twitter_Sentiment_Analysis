"""
Configuration file for the sentiment analysis project
This file stores all paths and hyperparameters in one place.
"""

import os  # used for handling file paths in a cross-platform way

# -------------------------------
# BASE DIRECTORY
# -------------------------------

# Gets the absolute path of the current file (config.py)
# Then moves up 2 levels to reach the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------
# DATA PATHS
# -------------------------------

# Main data folder
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# Raw data (original dataset)
RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw')

# Processed data (cleaned/modified dataset)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'Processed')

# -------------------------------
# OUTPUT DIRECTORIES
# -------------------------------

# Folder to save trained models
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# Folder to store results (metrics, plots, etc.)
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')

# -------------------------------
# DATASET PATH
# -------------------------------

# Full path to dataset file
DATASET_PATH = os.path.join(RAW_DATA_DIR, 'Twitter_Sentiment_Dataset.csv')

# -------------------------------
# MODEL PARAMETERS
# -------------------------------

# Fraction of data used for testing
TEST_SIZE = 0.2

# Random seed (ensures reproducibility)
RANDOM_STATE = 42

# Max number of features for TF-IDF
MAX_FEATURES = 5000

# -------------------------------
# CREATE DIRECTORIES (if missing)
# -------------------------------

# These lines ensure required folders exist
# If they already exist → nothing happens
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)