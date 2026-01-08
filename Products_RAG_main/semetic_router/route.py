from typing import List, Optional, Tuple
import numpy as np


# =========================================================
# ROUTE DEFINITION
# =========================================================
class Route:
    def __init__(self, name: str, samples: List[str]):
        self.name = name
        self.samples = samples


# =========================================================
# SEMANTIC ROUTER
# =========================================================
class SemanticRouter:
    def __init__(self, embedding=None, routes: List[Route] = None):
        """
        embedding:
          - ONLINE MODE : EmbeddingModel instance
          - OFFLINE MODE: None (benchmark)
        """
        self.routes = routes or []
        self.embedding = embedding
        self.routes_embedding = {}

        # ==============================
        # OFFLINE MODE (MT-RAG)
        # ==============================
        if self.embedding is None:
            return

        # ==============================
        # ONLINE MODE
        # ==============================
        for route in self.routes:
            all_embs = []
            batch_size = 64

            for i in range(0, len(route.samples), batch_size):
                batch = route.samples[i:i + batch_size]
                emb = self.embedding.encode(batch)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                all_embs.append(emb)

            self.routes_embedding[route.name] = np.vstack(all_embs)

    def guide(self, query: str) -> Tuple[float, str]:
        """
        Return (score, route_name)
        """

        # ==============================
        # OFFLINE MODE
        # ==============================
        if self.embedding is None:
            return 0.0, "clapnq"

        # ==============================
        # ONLINE MODE
        # ==============================
        query_emb = self.embedding.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores = []
        for route in self.routes:
            route_emb = self.routes_embedding.get(route.name)
            if route_emb is None:
                continue

            sims = np.dot(route_emb, query_emb.T).flatten()
            score = float(np.mean(sims))
            scores.append((score, route.name))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0]


# =========================================================
# ROUTE REGISTRY (4 COLLECTIONS)
# =========================================================
def build_default_router(embedding=None) -> SemanticRouter:
    routes = [
        Route(
            name="clapnq",
            samples=[
                "who is",
                "what is",
                "biography",
                "historical background",
                "general knowledge",
                "wikipedia article",
                "explain the concept",
                "background information",
            ],
        ),
        Route(
            name="govt",
            samples=[
                "government policy",
                "public administration",
                "law and regulation",
                "legal framework",
                "constitution",
                "public sector",
                "state authority",
                "government program",
            ],
        ),
        Route(
            name="fiqa",
            samples=[
                "finance",
                "economics",
                "stock market",
                "investment",
                "financial statement",
                "market cap",
                "interest rate",
                "economic analysis",
            ],
        ),
        Route(
            name="cloud",
            samples=[
                "cloud computing",
                "IBM Cloud",
                "AWS",
                "Azure",
                "Kubernetes",
                "DevOps",
                "infrastructure",
                "virtual machine",
            ],
        ),
    ]

    return SemanticRouter(
        embedding=embedding,
        routes=routes
    )
