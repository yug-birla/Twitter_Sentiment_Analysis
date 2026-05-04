import matplotlib.pyplot as plt

def plot_model_comparison(results_df):
    """
    Plot F1 scores for TF-IDF vs Word2Vec
    """

    # Separate features
    tfidf = results_df[results_df["feature"] == "TF-IDF"]
    w2v = results_df[results_df["feature"] == "Word2Vec"]

    plt.figure()

    # Plot TF-IDF
    plt.plot(tfidf["model"], tfidf["f1_score"], marker='o', label="TF-IDF")

    # Plot Word2Vec
    plt.plot(w2v["model"], w2v["f1_score"], marker='o', label="Word2Vec")

    plt.xlabel("Models")
    plt.ylabel("F1 Score")
    plt.title("TF-IDF vs Word2Vec Performance")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save plot
    plt.savefig("Results/model_comparison.png")

    plt.show()