"""
Model training functions
Handles training of different ML models
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    """
    Returns a dictionary of models to train
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }
    return models


def train_models(models, X_train, y_train):
    """
    Train all models and return trained versions
    """
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models