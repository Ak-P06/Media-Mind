import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import normalize


class ClusterEngine:

    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size

    def cluster(self, media_items):

        if not media_items:
            return []

        vectors = []

        for m in media_items:

            vec = m["vector"]

            # If embedder returned dict
            if isinstance(vec, dict):
                vec = vec.get("vector")

            if vec is None:
                vec = np.zeros(512)

            vec = np.array(vec, dtype=np.float32).flatten()

            if len(vec) != 512:
                vec = np.pad(vec, (0, 512 - len(vec))) if len(vec) < 512 else vec[:512]

            vectors.append(vec)   # ✅ FIXED INDENTATION

        vectors = np.stack(vectors)

        # Normalize embeddings
        vectors = normalize(vectors)

        n = len(vectors)

        # ---------- SELECT ALGORITHM ----------
        if n < 100:

            n_clusters = max(2, min(10, n // 10))

            model = AgglomerativeClustering(
                n_clusters=n_clusters
            )

        elif n < 500:

            model = DBSCAN(
                eps=0.25,
                min_samples=3,
                metric="cosine"
            )

        else:

            n_clusters = max(5, min(50, n // 20))

            model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )

        # ---------- RUN CLUSTERING ----------
        labels = model.fit_predict(vectors)

        print("Clusters found:", set(labels))

        result = []

        for i, item in enumerate(media_items):

            result.append({
                "path": item["path"],
                "cluster": int(labels[i])
            })

        return result