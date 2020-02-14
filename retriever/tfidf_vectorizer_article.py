import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


class TopTfIdf():
    def __init__(self, n_to_select: int, filter_dist_one: bool = False, rank=True):
        self.rank = rank
        self.n_to_select = n_to_select
        self.filter_dist_one = filter_dist_one

    def prune(self, question, paragraphs, return_scores=False):
        if not self.filter_dist_one and len(paragraphs) == 1:
            return paragraphs

        tfidf = TfidfVectorizer(strip_accents="unicode",
                                stop_words="english")
        text = []
        for para in paragraphs:
            text.append(para)
        try:
            para_features = tfidf.fit_transform(text)
        except ValueError:
            return []
        # question should be tokenized beforehand
        q_features = tfidf.transform([question])
        dists = cosine_similarity(q_features, para_features, "cosine").ravel()
        # in case of ties, use the earlier paragraph
        sorted_ix = np.argsort(dists)[::-1]
        if return_scores is True:
            return sorted_ix, dists
        else:
            return sorted_ix

    def dists(self, question, paragraphs):
        tfidf = TfidfVectorizer(strip_accents="unicode",
                                stop_words=self.stop.words)
        text = []
        for para in paragraphs:
            text.append(" ".join(" ".join(s) for s in para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        # in case of ties, use the earlier paragraph
        sorted_ix = np.lexsort(([x for x in range(len(paragraphs))], dists))

        if self.filter_dist_one:
            return [(paragraphs[i], dists[i]) for i in sorted_ix[:self.n_to_select] if dists[i] < 1.0]
        else:
            return [(paragraphs[i], dists[i]) for i in sorted_ix[:self.n_to_select]]
