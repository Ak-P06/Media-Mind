import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:

    @staticmethod
    def cosine(vec1, vec2):

        if vec1 is None or vec2 is None:
            return None

        v1 = np.asarray(vec1).reshape(1, -1)
        v2 = np.asarray(vec2).reshape(1, -1)

        return float(cosine_similarity(v1, v2)[0][0])

    @staticmethod
    def batch_cosine(query_vec, candidate_vecs):

        if query_vec is None or len(candidate_vecs) == 0:
            return []

        q = np.asarray(query_vec).reshape(1, -1)
        c = np.asarray(candidate_vecs)

        scores = cosine_similarity(q, c)[0]

        return scores.tolist()