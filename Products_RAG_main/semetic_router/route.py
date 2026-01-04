from typing import List, Optional, Tuple
import numpy as np


class Route:
    def __init__(self, name: Optional[str] = None, samples: Optional[List[str]] = None):
        self.name = name
        self.samples = samples or []


class SemanticRouter:
    def __init__(self, embedding=None, routes: List[Route] = None):
        """
        embedding:
          - ONLINE MODE : EmbeddingModel instance
          - OFFLINE MODE: None
        """
        self.routes = routes or []
        self.embedding = embedding
        self.routesEmbedding = {}

        # ==============================
        # OFFLINE MODE
        # ==============================
        if self.embedding is None:
            # Không khởi tạo embedding
            return

        # ==============================
        # ONLINE MODE
        # ==============================
        for route in self.routes:
            all_embs = []
            batch_size = 100

            for i in range(0, len(route.samples), batch_size):
                batch = route.samples[i:i + batch_size]
                emb = self.embedding.encode(batch)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                all_embs.append(emb)

            self.routesEmbedding[route.name] = np.vstack(all_embs)

    def get_routes(self):
        return self.routes

    def guide(self, query: str) -> Tuple[float, str]:
        """
        Return (score, route_name)
        """

        # ==============================
        # OFFLINE MODE
        # ==============================
        if self.embedding is None:
            # MT-RAG OFFLINE: luôn route về products
            return 0.0, "products"

        # ==============================
        # ONLINE MODE
        # ==============================
        queryEmbedding = self.embedding.encode([query])
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)

        scores = []
        for route in self.routes:
            emb = self.routesEmbedding.get(route.name)
            if emb is None:
                continue

            sims = np.dot(emb, queryEmbedding.T).flatten()
            score = float(np.mean(sims))
            scores.append((score, route.name))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0]
