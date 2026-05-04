# Twitter Sentiment Analysis: TF-IDF vs Word2Vec

## Overview

This project presents a complete end-to-end NLP pipeline for binary sentiment classification on Twitter data. The primary objective is a comparative study of two feature representation techniques :— TF-IDF and Word2Vec, across multiple machine learning models, with the goal of understanding how feature engineering choices impact classification performance.

---

## Objectives

- Build a modular, reproducible NLP pipeline from scratch
- Conduct structured Exploratory Data Analysis (EDA)
- Compare TF-IDF and Word2Vec as feature representations
- Evaluate multiple ML classifiers under consistent conditions
- Derive data-backed conclusions on the relationship between feature choice and model performance

---

## Project Structure

```
Twitter Sentiment Analysis/
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── Models/
│
├── Results/
│   ├── final_comparison.csv
│   └── model_comparison.png
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** Twitter Sentiment Dataset (Sentiment140-style)
- **Size:** ~100,000 tweets
- **Labels:**
  - `0` — Negative sentiment
  - `4` — Positive sentiment (remapped to `1` during preprocessing)

**Final columns used:**

| Column | Description |
|---|---|
| `text` | Raw tweet content |
| `label` | Binary sentiment label (0 or 1) |
| `cleaned_text` | Preprocessed tweet text |

---

## Exploratory Data Analysis

Full analysis is available in `notebooks/EDA.ipynb`.

**Key findings:**

- **Balanced classes:** Approximately 50% positive and 50% negative samples; no resampling was required.
- **Short text length:** Tweets are brief with limited context, which reduces the advantage of semantic embedding models.
- **Strong sentiment keywords:** Positive tweets feature words such as *love*, *great*, and *happy*; negative tweets feature words such as *bad*, *sad*, and *problem*.
- **Repetitive vocabulary:** High word frequency provides a strong signal for frequency-based features like TF-IDF.
- **Weak contextual dependency:** The absence of long-range semantic structure limits the utility of Word2Vec representations.

---

## Text Preprocessing

Implemented in `preprocessing.py`.

**Pipeline steps:**

1. Lowercasing
2. URL removal
3. Mention removal (`@user`)
4. Hashtag cleaning
5. Special character removal
6. Stopword removal with negation preservation
7. Lemmatization

---

## Feature Engineering

### TF-IDF (Term Frequency–Inverse Document Frequency)

- Produces sparse, high-dimensional vectors (5,000 features)
- Captures word importance relative to the corpus
- Well-suited for short text with strong keyword signals

### Word2Vec (Neural Word Embeddings)

- Learns dense word embeddings from the training corpus
- Sentence-level representation computed as the average of constituent word vectors
- Captures semantic similarity between words

---

## Models

The following classifiers were evaluated:

| Model | TF-IDF | Word2Vec |
|---|---|---|
| Logistic Regression | Yes | Yes |
| Support Vector Machine (SVM) | Yes | Yes |
| Random Forest | Yes | No |
| XGBoost | Yes | No |

> Random Forest and XGBoost were excluded from the Word2Vec evaluation due to performance and efficiency considerations with dense embeddings.

---

## Results

### Final Model Comparison

| Model | Feature | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | TF-IDF | **0.7665** | **0.7697** |
| SVM | TF-IDF | 0.7609 | 0.7644 |
| Random Forest | TF-IDF | 0.7513 | 0.7441 |
| XGBoost | TF-IDF | 0.7447 | 0.7597 |
| Logistic Regression | Word2Vec | 0.7178 | 0.7219 |
| SVM | Word2Vec | 0.7182 | 0.7235 |

The best-performing configuration is **Logistic Regression with TF-IDF**, saved to `Models/logistic_tfidf.pkl`.

Visualization of results is available at `Results/model_comparison.png`.

---

## Key Insights

**1. TF-IDF outperforms Word2Vec on this dataset.**
TF-IDF preserves word importance and frequency signals. Word2Vec's averaged sentence vectors dilute intensity and lose negation context (e.g., "not good").

**2. Linear models are most effective.**
Logistic Regression and SVM outperform tree-based ensembles, indicating that the feature space is largely linearly separable.

**3. Feature engineering has greater impact than model complexity.**
Simpler models paired with well-constructed features consistently outperform more complex models with weaker representations.

**4. EDA findings are predictive of model behavior.**

| EDA Finding | Model Implication |
|---|---|
| Short tweets | Linear models are sufficient |
| Strong keywords | TF-IDF captures the relevant signal |
| Repetitive vocabulary | Frequency-based features are effective |
| Weak contextual structure | Word2Vec embeddings offer limited benefit |

---

## How to Run

```bash
# Activate virtual environment
venv311\Scripts\activate       # Windows
source venv311/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

---

## Tech Stack

- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Gensim
- Matplotlib, Seaborn
- NLTK

---

## Conclusion

For short-text sentiment classification, TF-IDF combined with linear classifiers represents a strong and interpretable baseline. This project demonstrates that careful feature engineering can consistently outperform more complex representation learning approaches on tasks where semantic context is limited and keyword signals are strong.
