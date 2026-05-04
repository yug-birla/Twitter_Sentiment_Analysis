"""
Text preprocessing for Twitter sentiment analysis
This module cleans and prepares raw tweets for ML models
"""

import re  # for regex-based text cleaning
import nltk  # NLP library for tokenization, stopwords, etc.

# Import specific NLP tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd


# -------------------------------
# DOWNLOAD REQUIRED NLTK DATA
# -------------------------------

# These resources are needed for tokenization, stopwords, and lemmatization
# This block ensures they are downloaded only once


def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
# -------------------------------
# MAIN PREPROCESSOR CLASS
# -------------------------------

class TweetPreprocessor:
    """Handles full tweet preprocessing pipeline"""

    def __init__(self):
        # Load English stopwords (common words with low meaning)
        self.stop_words = set(stopwords.words('english'))

        # Initialize lemmatizer (reduces words to base form)
        self.lemmatizer = WordNetLemmatizer()

        # IMPORTANT: Keep negation words (they affect sentiment)
        # Example: "not good" vs "good"
        self.stop_words -= {
            'not', 'no', 'nor', 'neither', 'never',
            'none', 'nothing', 'nobody'
        }

    # -------------------------------
    # STEP 1: BASIC CLEANING
    # -------------------------------

    def clean_tweet(self, text):
        """Remove noise from raw tweet"""

        # Convert to lowercase → ensures consistency
        text = text.lower()

        # Remove URLs (http, https, www links)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions like @username
        text = re.sub(r'@\w+', '', text)

        # Remove '#' but keep the word (e.g., #happy → happy)
        text = re.sub(r'#', '', text)

        # Remove numbers and special characters
        # Keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # -------------------------------
    # STEP 2: TOKENIZATION + LEMMATIZATION
    # -------------------------------

    def tokenize_and_lemmatize(self, text):
        """Convert text → tokens → cleaned tokens"""

        # Split sentence into words
        tokens = word_tokenize(text)

        # Process each word:
        # - remove stopwords
        # - remove very short words (len <= 2)
        # - convert to base form (running → run)
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        # Join tokens back into sentence
        return ' '.join(tokens)

    # -------------------------------
    # FULL PIPELINE (CLEAN + TOKENIZE)
    # -------------------------------

    def preprocess(self, text):
        """Apply full preprocessing to a single tweet"""

        text = self.clean_tweet(text)                # remove noise
        text = self.tokenize_and_lemmatize(text)     # normalize words

        return text

    # -------------------------------
    # APPLY TO ENTIRE DATAFRAME
    # -------------------------------

    def preprocess_dataframe(self, df, text_column='text'):
        """Apply preprocessing to entire dataset"""

        print("Preprocessing tweets...")

        # Apply preprocessing to each tweet
        df['cleaned_text'] = df[text_column].apply(self.preprocess)

        # Remove rows where text became empty after cleaning
        original_len = len(df)

        df = df[df['cleaned_text'].str.len() > 0]

        removed = original_len - len(df)

        print("Preprocessing complete!")
        print(f"Removed {removed} empty tweets")
        print(f"Remaining tweets: {len(df)}")

        return df


# -------------------------------
# DEBUGGING / VISUALIZATION
# -------------------------------

def show_cleaning_examples(df, n=5):
    """Show before vs after cleaning"""

    print("\n" + "=" * 80)
    print("CLEANING EXAMPLES")
    print("=" * 80)

    for i in range(min(n, len(df))):
        print(f"\n[{i+1}] ORIGINAL:")
        print(f"    {df.iloc[i]['text']}")

        print("    CLEANED:")
        print(f"    {df.iloc[i]['cleaned_text']}")

    print("=" * 80)