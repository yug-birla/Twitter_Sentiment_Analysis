"""
Model evaluation functions
Handles performance metrics and comparison
"""

from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

import numpy as np


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models and return results
    """
    results = []

    for name, model in models.items():
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_score": f1
        })
    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df