"""
Main script to run the sentiment analysis pipeline
"""

# -------------------------------
# IMPORTS
# -------------------------------
from src.utils import load_dataset, get_data_info
from src.preprocessing import TweetPreprocessor, show_cleaning_examples, download_nltk_resources
from src.features import get_tfidf_features, train_word2vec, get_word2vec_features
from src.train import get_models, train_models
from src.evaluate import evaluate_models
from src import config
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.visualize import plot_model_comparison

def main():
    print("Starting Twitter Sentiment Analysis Project\n")

    # -------------------------------
    # STEP 1: LOAD DATA
    # -------------------------------
    print("Loading dataset...")
    df = load_dataset()

    if df is None:
        print("Dataset loading failed.")
        return

    # -------------------------------
    # STEP 2: DATA INSPECTION
    # -------------------------------
    get_data_info(df)

    # -------------------------------
    # STEP 3: TEXT PREPROCESSING
    # -------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: TEXT PREPROCESSING")
    print("=" * 50)

    download_nltk_resources()

    preprocessor = TweetPreprocessor()
    df = preprocessor.preprocess_dataframe(df)

    show_cleaning_examples(df, n=5)

    # -------------------------------
    # STEP 4: SAVE PROCESSED DATA
    # -------------------------------
    processed_path = f"{config.PROCESSED_DATA_DIR}/processed_tweets.csv"
    df.to_csv(processed_path, index=False)

    print(f"\nProcessed data saved to: {processed_path}")

    # -------------------------------
    # STEP 5: TRAIN-TEST SPLIT
    # -------------------------------
    print("\n" + "=" * 50)
    print("STEP 5: TRAIN-TEST SPLIT")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # =========================================================
    # TF-IDF PIPELINE
    # =========================================================
    print("\n" + "=" * 50)
    print("STEP 6: TF-IDF FEATURE EXTRACTION")
    print("=" * 50)

    X_train_tfidf, X_test_tfidf, _ = get_tfidf_features(
        X_train, X_test, config.MAX_FEATURES
    )

    print(f"TF-IDF shape: {X_train_tfidf.shape}")

    print("\n--- Training TF-IDF Models ---")
    tfidf_models = train_models(get_models(), X_train_tfidf, y_train)

    print("\n--- Evaluating TF-IDF Models ---")
    tfidf_results = evaluate_models(tfidf_models, X_test_tfidf, y_test)
    tfidf_results["feature"] = "TF-IDF"

    # =========================================================
    # WORD2VEC PIPELINE
    # =========================================================
    print("\n" + "=" * 50)
    print("STEP 7: WORD2VEC FEATURE EXTRACTION")
    print("=" * 50)

    # Tokenization
    X_train_tokens = X_train.apply(lambda x: x.split())
    X_test_tokens = X_test.apply(lambda x: x.split())

    # Train Word2Vec
    w2v_model = train_word2vec(X_train_tokens, vector_size=100)

    # Convert sentences → vectors
    X_train_w2v = get_word2vec_features(X_train_tokens, w2v_model)
    X_test_w2v = get_word2vec_features(X_test_tokens, w2v_model)

    print(f"Word2Vec shape: {X_train_w2v.shape}")

    # -------------------------------
    # IMPORTANT: Only fast models
    # -------------------------------
    print("\n--- Training Word2Vec Models ---")

    all_models = get_models()

    w2v_models_dict = {
        "Logistic Regression": all_models["Logistic Regression"],
        "SVM": all_models["SVM"]
    }

    w2v_models = train_models(w2v_models_dict, X_train_w2v, y_train)

    print("\n--- Evaluating Word2Vec Models ---")
    w2v_results = evaluate_models(w2v_models, X_test_w2v, y_test)
    w2v_results["feature"] = "Word2Vec"

    # =========================================================
    # FINAL COMPARISON
    # =========================================================
    print("\n" + "=" * 50)
    print("FINAL COMPARISON: TF-IDF vs Word2Vec")
    print("=" * 50)

    final_results = pd.concat([tfidf_results, w2v_results])

    print(final_results)

    # Save results
    results_path = f"{config.RESULTS_DIR}/final_comparison.csv"
    final_results.to_csv(results_path, index=False)

    print(f"\nFinal comparison saved to: {results_path}")

    best_model = tfidf_models["Logistic Regression"]

    model_path = f"{config.MODELS_DIR}/logistic_tfidf.pkl"
    joblib.dump(best_model, model_path)

    print(f"Model saved at: {model_path}")
    
    plot_model_comparison(final_results)
if __name__ == "__main__":
    main()