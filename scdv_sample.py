from scdv import SCDVVectorizer

if __name__ == "__main__":

    docs = ["わたし は かもめ",
            "かもめ の ジョナサン",
            "ファミレス と いう ば ジョナサン"]
    vectorizer = SCDVVectorizer(docs, window=1, min_count=1)
    vecs = vectorizer.fit_transform(docs)
    print(vecs[0])
    vectorizer.save("./sample_vectorizer.model")

    loaded_vectorizer = SCDVVectorizer.load("./sample_vectorizer.model")
    new_vec = loaded_vectorizer.transform(["かもめ は かもめ"])
    print(new_vec)
