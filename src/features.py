from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def get_tfidf_features(train_text, test_text, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    return X_train, X_test, vectorizer


def train_word2vec(tokenized_texts, vector_size):
    """
    Train Word2Vec model on tokenized sentences
    """
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4
    )
    return model


def get_sentence_vector(tokens, model):
    """
    Convert a sentence (list of words) into a vector
    by averaging word vectors
    """
    vectors = []

    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    # If no known words → return zero vector
    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def get_word2vec_features(tokenized_texts, model):
    """
    Convert list of tokenized sentences into feature matrix
    """
    features = []

    for tokens in tokenized_texts:
        vec = get_sentence_vector(tokens, model)
        features.append(vec)

    return np.array(features)