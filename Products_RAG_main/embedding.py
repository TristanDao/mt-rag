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
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()

        # ======================================================
        # OPENAI EMBEDDINGS
        # ======================================================
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed")

            self.client = OpenAI(api_key=self.api_key)
            self.model_name = model_name or "text-embedding-3-small"

            # 🔴 QUAN TRỌNG: expose vector_size
            self.vector_size = self._infer_openai_vector_size()

        # ======================================================
        # GEMINI (DISABLED)
        # ======================================================
        elif self.provider == "gemini":
            raise RuntimeError(
                "Gemini Embedding hiện không được hỗ trợ.\n"
                "Hãy dùng provider='huggingface' hoặc 'openai'."
            )

        # ======================================================
        # HUGGINGFACE LOCAL EMBEDDINGS
        # ======================================================
        elif self.provider == "huggingface":
            self.model_name = config.EMBEDDING_MODEL_CONFIG.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
            device = "cuda" if torch.cuda.is_available() else "cpu"
            local_model_path = os.path.join("models", os.path.basename(self.model_name))

            if not os.path.exists(local_model_path):
                print(f"Tải model từ Hugging Face: {self.model_name}")
                model = SentenceTransformer(self.model_name, device=device)
                os.makedirs("models", exist_ok=True)
                model.save(local_model_path)
                print(f"Model đã lưu tại: {local_model_path}")

            print(f"Load model từ local: {local_model_path}")
            self.model = SentenceTransformer(self.model_name, device=device)

            # 🔴 QUAN TRỌNG: expose vector_size
            self.vector_size = self.model.get_sentence_embedding_dimension()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------

    def _get_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        return None

    # ------------------------------------------------------
    # OPENAI VECTOR SIZE INFERENCE
    # ------------------------------------------------------
    def _infer_openai_vector_size(self) -> int:
        if not self.model_name:
            return 1536
        if "text-embedding-3-large" in self.model_name:
            return 3072
        if "text-embedding-3-small" in self.model_name:
            return 1536
        return 1536

    # ------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ------------------------------------------------------
    def get_vector_size(self) -> int:
        """
        Giữ lại cho code cũ / evaluator
        """
        return self.vector_size

    # ------------------------------------------------------

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            raise ValueError("Input must be string or list[str]")

        # ==========================
        # OPENAI
        # ==========================
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [d.embedding for d in response.data]

        # ==========================
        # HUGGINGFACE
        # ==========================
        elif self.provider == "huggingface":
            return self.model.encode(
                texts,
                convert_to_tensor=False
            ).tolist()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------------------------------

    def encode_single(self, text: str) -> List[float]:
        result = self.encode(text)
        return result[0] if result else []


# ==========================================================
# SPARSE EMBEDDING (BM25 / FASTEMBED)
# ==========================================================
class SparseEmbeddingModel:
    def __init__(
        self,
        provider: str = "fastembed",
        model_name: str = "Qdrant/bm25"
    ):
        self.provider = provider
        self.model_name = model_name

        if self.provider == "fastembed":
            try:
                from fastembed import SparseTextEmbedding
                print(f"Loading Sparse Model: {self.model_name}")
                self.model = SparseTextEmbedding(model_name=self.model_name)
            except ImportError:
                raise ImportError("Please install fastembed: pip install fastembed")
        else:
            raise ValueError(f"Unsupported sparse provider: {self.provider}")

    def encode(
        self,
        texts: Union[str, List[str]]
    ) -> List[Dict[str, Union[List[int], List[float]]]]:
        if isinstance(texts, str):
            texts = [texts]

        results = []
        embeddings = self.model.embed(texts)

        for sparse_vec in embeddings:
            results.append({
                "indices": sparse_vec.indices.tolist(),
                "values": sparse_vec.values.tolist()
            })

        return results

    def encode_single(
        self,
        text: str
    ) -> Dict[str, Union[List[int], List[float]]]:
        return self.encode(text)[0]