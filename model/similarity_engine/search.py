import numpy as np
from .similarity import SimilarityEngine


class SearchEngine:

    def __init__(self, embeddings):
        """
        embeddings = {path: vector}
        """

        vectors = []
        paths = []

        for p, v in embeddings.items():

            if v is None:
                continue

            if isinstance(v, dict):
                v = v.get("vector")

            vec = np.array(v, dtype=np.float32).flatten()

            if len(vec) != 512:
                vec = np.pad(vec, (0, 512 - len(vec))) if len(vec) < 512 else vec[:512]

            vectors.append(vec)
            paths.append(p)

        self.paths = paths
        self.matrix = np.stack(vectors)
        self.matrix = self.matrix / (np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-8)

    def search(self, query_vec, top_k=5):

        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        scores = SimilarityEngine.batch_cosine(query_vec, self.matrix)

        ranked = sorted(
            zip(self.paths, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
    
    def text_similarity(self, text):
        return {"message": "Text search not implemented yet"}

    def image_similarity(self, image_data):
        return {"message": "Image search not implemented yet"}