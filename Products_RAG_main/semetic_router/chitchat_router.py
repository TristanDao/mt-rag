# semetic_router/chitchat_router.py

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChitchatRouter:
    def __init__(
        self,
        samples: list[str],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.samples = samples
        self.sample_embeddings = self.model.encode(
            samples,
            normalize_embeddings=True
        )

    def similarity(self, query: str) -> float:
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        sims = cosine_similarity(q_emb, self.sample_embeddings)
        return float(np.max(sims))

    def is_chitchat(self, query: str, threshold: float = 0.50) -> bool:
        return self.similarity(query) >= threshold
