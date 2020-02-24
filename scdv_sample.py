from scdv import SCDVVectorizer
from yellowbrick.datasets import load_hobbies
from yellowbrick.text import TSNEVisualizer

# from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":

    corpus = load_hobbies()
    vectorizer = SCDVVectorizer(
        # w2v_model_path="word2vec.gensim.model",
        size=400, n_components=10, iter=200, sparsity_percentage=0.1)
    X = vectorizer.fit_transform(corpus.data)
    y = corpus.target
    # vectorizer.save("./sample_vectorizer.pkl")
    # loaded_vectorizer = SCDVVectorizer.load("./sample_vectorizer.pkl")
    # X = loaded_vectorizer.transform(corpus)

    # Create the visualizer and draw the vectors
    tsne = TSNEVisualizer(decompose_by=20,
                          random_state=42,
                          perplexity=15,
                          learning_rate=200.0,
                          n_iter=10000)
    tsne.fit(X, y)
    tsne.show()
