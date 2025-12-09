from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import os
from typing import List, Union, Optional
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    def __init__(self, provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()

        # ==========================================================
        # OPENAI EMBEDDINGS
        # ==========================================================
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = model_name or "text-embedding-3-small"

        # ==========================================================
        # ❌ GEMINI EMBEDDINGS – TẠM THỜI VÔ HIỆU ĐỂ TRÁNH CRASH
        # ==========================================================
        elif self.provider == "gemini":
            raise RuntimeError(
                "Gemini Embedding hiện không được hỗ trợ trong evaluator.\n"
                "Hãy dùng provider='huggingface' hoặc 'openai'."
            )

        # ==========================================================
        # HUGGINGFACE LOCAL EMBEDDINGS (KHÔNG DÙNG INTERNET)
        # ==========================================================
        elif self.provider == "huggingface":
            self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

            device = "cuda" if torch.cuda.is_available() else "cpu"
            local_model_path = os.path.join("models", os.path.basename(self.model_name))

            # Nếu model chưa tồn tại → tải về
            if not os.path.exists(local_model_path):
                print(f"Tải model từ Hugging Face: {self.model_name}")
                model = SentenceTransformer(self.model_name, device=device)
                os.makedirs("models", exist_ok=True)
                model.save(local_model_path)
                print(f"Model đã lưu tại: {local_model_path}")

            print(f"Load model từ local: {local_model_path}")
            self.model = SentenceTransformer(self.model_name, device=device)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ----------------------------------------------------------------------

    def _get_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        return None

    # ----------------------------------------------------------------------

    def get_vector_size(self) -> int:
        if self.provider == "openai":
            if "text-embedding-3-small" in self.model_name:
                return 1536
            if "text-embedding-3-large" in self.model_name:
                return 3072
            return 1536

        elif self.provider == "huggingface":
            return self.model.get_sentence_embedding_dimension()

        elif self.provider == "gemini":
            # Nếu bật lại Gemini embedding
            return 3072

        return 1536

    # ----------------------------------------------------------------------

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            raise ValueError("Input must be string or list[str]")

        # ======================================
        # OPENAI
        # ======================================
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [d.embedding for d in response.data]

        # ======================================
        # HUGGINGFACE
        # ======================================
        elif self.provider == "huggingface":
            return self.model.encode(texts, convert_to_tensor=False).tolist()

        # ======================================
        # GEMINI – hiện vô hiệu
        # ======================================
        elif self.provider == "gemini":
            raise RuntimeError("Gemini embedding disabled.")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ----------------------------------------------------------------------

    def encode_single(self, text: str) -> List[float]:
        result = self.encode(text)
        return result[0] if result else []
