from gensim.models.word2vec import Word2Vec, MAX_WORDS_IN_BATCH

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import BaseEstimator, TransformerMixin, TfidfVectorizer
from sklearn.mixture import GaussianMixture

import numpy as np


class SCDVVectorizer(Word2Vec, BaseEstimator, TransformerMixin):
    @staticmethod
    def tokenize_simply(text):
        return text.split()

    def __init__(self,
                 sentences=None,
                 size=100,
                 alpha=0.025,
                 window=5,
                 min_count=5,
                 max_vocab_size=None,
                 sample=0.001,
                 seed=1,
                 workers=3,
                 min_alpha=0.0001,
                 sg=0,
                 hs=0,
                 negative=5,
                 cbow_mean=1,
                 hashfxn=hash,
                 iter=5,
                 null_word=0,
                 trim_rule=None,
                 sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH,
                 compute_loss=False,
                 callbacks=(),
                 n_components=1,
                 covariance_type="full",
                 tol=0.001,
                 reg_covar=1e-06,
                 max_iter=100,
                 n_init=1,
                 init_params="kmeans",
                 weights_init=None,
                 means_init=None,
                 precisions_init=None,
                 random_state=None,
                 warm_start=False,
                 verbose=0,
                 verbose_interval=10,
                 input="content",
                 encoding="utf-8",
                 decode_error="strict",
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 analyzer="word",
                 stop_words=None,
                 token_pattern=r"(?u)\b\w\w +\b",
                 ngram_range=(1, 1),
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.float64,
                 norm="l2",
                 use_idf=True,
                 smooth_idf=True,
                 sublinear_tf=False):

        if tokenizer is None:
            tokenizer = SCDVVectorizer.tokenize_simply

        sentences = [tokenizer(sentence) for sentence in sentences]

        super().__init__(
            sentences=sentences,
            size=size,
            alpha=alpha,
            window=window,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            sample=sample,
            seed=seed,
            workers=workers,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            negative=negative,
            cbow_mean=cbow_mean,
            hashfxn=hashfxn,
            iter=iter,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            batch_words=batch_words,
            compute_loss=compute_loss,
            callbacks=callbacks)

        self.tokenizer = tokenizer

        self.gmm_params = {"n_components": n_components,
                           "covariance_type": covariance_type,
                           "tol": tol,
                           "reg_covar": reg_covar,
                           "max_iter": max_iter,
                           "n_init": n_init,
                           "init_params": init_params,
                           "weights_init": weights_init,
                           "means_init": means_init,
                           "precisions_init": precisions_init,
                           "random_state": random_state,
                           "warm_start": warm_start,
                           "verbose": verbose,
                           "verbose_interval": verbose_interval}

        self.tfidf_params = {"input": input,
                             "encoding": encoding,
                             "decode_error": decode_error,
                             "strip_accents": strip_accents,
                             "lowercase": lowercase,
                             "preprocessor": preprocessor,
                             "tokenizer": tokenizer,
                             "analyzer": analyzer,
                             "stop_words": stop_words,
                             "token_pattern": token_pattern,
                             "ngram_range": ngram_range,
                             "max_df": max_df,
                             "min_df": min_df,
                             "max_features": max_features,
                             "vocabulary": vocabulary,
                             "binary": binary,
                             "dtype": dtype,
                             "norm": norm,
                             "use_idf": use_idf,
                             "smooth_idf": smooth_idf,
                             "sublinear_tf": sublinear_tf}

        self.gmm = GaussianMixture(**self.gmm_params)
        self.n_components = n_components
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        self._fitted = False

    def __idf_fit_tranform(self, X):
        self.tfidf_vectorizer.fit(X)
        vocab = self.tfidf_vectorizer.get_feature_names()
        idf_values = self.tfidf_vectorizer._tfidf.idf_
        return set(vocab), idf_values

    def __gmm_fit_predict(self):
        wv = []
        for w in self.wv.vocab:
            wv.append(self.wv[w])
        clusters = self.gmm.fit_predict(wv)
        clust_probas = self.gmm.predict_proba(wv)
        return set(self.wv.vocab), clusters, clust_probas

    def __get_word_dict(self, vocab, values):
        word_dict = dict()
        for w, value in zip(vocab, values):
            word_dict[w] = value

        return word_dict

    def __check_input(self, raw_documents) -> bool:
        if isinstance(raw_documents, str):
            raise TypeError(
                "The type of raw_documents must be iterable. Not str.")
        else:
            return True

    def fit_transform(self, raw_documents, y=None):
        self.__check_input(raw_documents)
        sentences = [self.tokenizer(sentence) for sentence in raw_documents]
        # Obtain word vector
        self.train(sentences, total_examples=len(
            sentences), epochs=self.epochs)
        # Calculate idf values
        vocab, idf_values = self.__idf_fit_tranform(raw_documents)
        # Cluster word vectors using GMM and
        # obtain soft assignment P(cluster_k|word_i)
        gmm_vocab, gmm_clusters, gmm_clust_probas = self.__gmm_fit_predict()

        assert vocab == vocab & gmm_vocab

        # for each word_i in vocabulary do
        word_clust_prob_dict = self.__get_word_dict(gmm_vocab,
                                                    gmm_clust_probas)
        word_idf_dict = self.__get_word_dict(vocab, idf_values)
        tmp_wtvec_dict = dict()
        for w in vocab:
            wcvec = list()
            clust_prob_array = word_clust_prob_dict[w]
            for p_ck_wi in clust_prob_array:
                # Calc wordvec_i * P(cluster_k|word_i) as wcvec_ik
                wcvec_ik = self.wv[w] * p_ck_wi
                wcvec.append(wcvec_ik)

            # Calc idf(word_i) * concat(wcvec_i1 to wcvec_ik) as wtvec_i
            idf_wi = word_idf_dict[w]
            concat_vec = np.hstack(wcvec)
            wtvec_i = idf_wi * concat_vec
            tmp_wtvec_dict[w] = wtvec_i

        self.wtvec_dict_ = tmp_wtvec_dict
        self._fitted = True
        return self.transform(raw_documents)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents):
        if not self._fitted:
            raise NotFittedError(
                "This SCDVVectorizer instance is not fitted yet.")

        self.__check_input(raw_documents)
        sentences = [self.tokenizer(sentence) for sentence in raw_documents]
        docvecs = []
        # for each document_n in X do
        for sentence in sentences:
            # init docvec_n as zero vectors
            docvec_n = np.zeros(self.wv.vector_size * self.n_components)
            # for each word_i in document_n do
            for w in sentence:
                wtvec_i = self.wtvec_dict_[w]
                docvec_n += wtvec_i

            docvecs.append(docvec_n)
        return np.array(docvecs)
