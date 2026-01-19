try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from sentence_transformers import SentenceTransformer
import torch
import os
from typing import List, Union, Optional, Dict
from dotenv import load_dotenv
import config

load_dotenv()

# ==========================================================
# DENSE EMBEDDING MODEL
# ==========================================================
class EmbeddingModel:
    def __init__(
        self,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.model_name = model_name or config.EMBEDDING_MODEL_CONFIG.get("model_name")
        self.api_key = api_key or self._get_api_key()

        # ==========================
        # OPENAI
        # ==========================
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed")

            self.client = OpenAI(api_key=self.api_key)
            self.model_name = self.model_name or "text-embedding-3-small"
            self.vector_size = self._infer_openai_vector_size()

        # ==========================
        # HUGGINGFACE
        # ==========================
        elif self.provider == "huggingface":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔹 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=device)
            self.vector_size = self.model.get_sentence_embedding_dimension()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------
    def _get_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return None

    # ------------------------------------------------------
    def _infer_openai_vector_size(self) -> int:
        if "large" in self.model_name:
            return 3072
        return 1536

    # ======================================================
    # 🔴 E5 PREFIX HANDLING (CORE FIX)
    # ======================================================
    def _add_prefix(self, text: str, mode: str) -> str:
        """
        mode: 'query' | 'passage'
        """
        name = self.model_name.lower()

        if "e5" not in name:
            return text

        # E5-instruct or Qwen (Instruction Aware)
        if "instruct" in name or "qwen" in name:
            if mode == "query":
                # Default retrieval instruction
                task_description = "Given a web search query, retrieve relevant passages that answer the query"
                return f"Instruct: {task_description}\nQuery: {text}"
            else:
                # Passage needs NO prefix for E5-instruct
                return text

        # E5 base / large (non-instruct)
        prefix = "query:" if mode == "query" else "passage:"
        return f"{prefix} {text}"

    # ======================================================
    # MAIN ENCODE
    # ======================================================
    def encode(
        self,
        texts: Union[str, List[str]],
        mode: str = "passage",
        **kwargs
    ) -> List[List[float]]:

        if isinstance(texts, str):
            texts = [texts]

        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [d.embedding for d in response.data]

        elif self.provider == "huggingface":
            texts = [self._add_prefix(t, mode) for t in texts]
            return self.model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                **kwargs
            ).tolist()

        else:
            raise ValueError("Unsupported provider")

    # ======================================================
    # PUBLIC APIS
    # ======================================================
    def encode_query(self, text: str) -> List[float]:
        return self.encode(text, mode="query")[0]

    def encode_passage(self, text: str) -> List[float]:
        return self.encode(text, mode="passage")[0]

    # ------------------------------------------------------
    # BACKWARD COMPAT (GIỮ CHO CODE CŨ)
    # ------------------------------------------------------
    def encode_single(self, text: str) -> List[float]:
        """
        ⚠️ Legacy: mặc định coi là query
        """
        return self.encode_query(text)

    def get_vector_size(self) -> int:
        return self.vector_size


# ==========================================================
# SPARSE EMBEDDING (BM25)
# ==========================================================
class SparseEmbeddingModel:
    def __init__(
        self,
        provider: str = "fastembed",
        model_name: str = "Qdrant/bm25"
    ):
        self.provider = provider
        self.model_name = model_name

        if provider == "fastembed":
            from fastembed import SparseTextEmbedding
            self.model = SparseTextEmbedding(model_name=self.model_name)
        else:
            raise ValueError("Unsupported sparse provider")

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for vec in self.model.embed(texts):
            vectors.append({
                "indices": vec.indices.tolist(),
                "values": vec.values.tolist()
            })
        return vectors

    def encode_single(self, text: str):
        return self.encode(text)[0]
