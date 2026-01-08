from enum import Enum
from typing import List, Dict, Any, Optional
import os
import sys
from dotenv import load_dotenv

from openai import AzureOpenAI

# =========================================================
# PATH SETUP
# Products_RAG_main/.. -> PROJECT ROOT
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# SAFE IMPORTS (PROJECT ROOT)
# =========================================================
from response_generator import ResponseGenerator

load_dotenv()


# =========================================================
# MODE
# =========================================================
class RAGMode(str, Enum):
    ONLINE = "online"     # App / Streamlit
    OFFLINE = "offline"   # MT-RAG benchmark


# =========================================================
# RAG SYSTEM (CORE – STATELESS)
# =========================================================
class RAGSystem:
    def __init__(
        self,
        use_vector_db: bool = False,
        db_type: str = "qdrant",
        embedding_provider: str = "huggingface",
        embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        collection_name: Optional[str] = None,   # ✅ KHÔNG HARD-CODE
    ):
        """
        RAG core system.
        - KHÔNG quyết định collection
        - KHÔNG routing
        - KHÔNG reflection
        """

        self.use_vector_db = use_vector_db
        self.collection_name = collection_name

        # =================================================
        # VECTOR DB + EMBEDDING (OPTIONAL)
        # =================================================
        if self.use_vector_db:
            from Products_RAG_main.vector_db import VectorDatabase
            from Products_RAG_main.embedding import EmbeddingModel

            print("[RAG] Initializing VectorDB + Embedding")

            self.vector_db = VectorDatabase(db_type=db_type)

            self.embedding_model = EmbeddingModel(
                provider=embedding_provider,
                model_name=embedding_model
            )

            # ❗ Không tạo collection nếu chưa được set từ ngoài
            if self.collection_name:
                self.vector_db.create_collection_if_not_exists(
                    collection_name=self.collection_name,
                    vector_size=self.embedding_model.get_vector_size()
                )
        else:
            self.vector_db = None
            self.embedding_model = None
            print("[RAG] VectorDB disabled (LLM-only mode)")

        # =================================================
        # LLM (AZURE OPENAI)
        # =================================================
        if llm_provider.lower() != "openai":
            raise ValueError("Only OpenAI/Azure OpenAI is supported")

        self.llm_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        self.llm_model = llm_model or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        print("[RAG] LLM initialized")

        # =================================================
        # RESPONSE GENERATOR
        # =================================================
        self.response_generator = ResponseGenerator(
            temperature=0.0,
            max_tokens=800,
        )

        print("[RAG] System ready\n")

    # =====================================================
    # RETRIEVE (ONLINE + VECTOR DB)
    # =====================================================
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if (
            not self.use_vector_db
            or self.vector_db is None
            or not self.collection_name
        ):
            return []

        query_embedding = self.embedding_model.encode_single(query)

        return self.vector_db.query(
            collection_name=self.collection_name,
            embedding_vector=query_embedding,
            top_k=top_k
        )

    # =====================================================
    # FORMAT CONTEXT
    # =====================================================
    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""

        chunks = []
        for i, doc in enumerate(docs):
            payload = doc.get("payload", {})
            text = payload.get("text") or payload.get("content") or ""
            if text:
                chunks.append(f"Tài liệu {i + 1}:\n{text}")

        return "\n\n".join(chunks)

    # =====================================================
    # QUERY ENTRYPOINT
    # =====================================================
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        mode: RAGMode = RAGMode.ONLINE,
        provided_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        # =================================================
        # OFFLINE MODE (BENCHMARK)
        # =================================================
        if mode == RAGMode.OFFLINE:
            if not provided_contexts:
                raise ValueError("OFFLINE mode requires provided_contexts")

            context = self.format_context(provided_contexts)

            answer = self.response_generator.generate(
                prompt_type="standard",
                query=user_query,
                context=context,
            )

            return {
                "answer": answer,
                "context": context,
                "documents": provided_contexts,
                "mode": "offline",
            }

        # =================================================
        # ONLINE MODE
        # =================================================
        if mode == RAGMode.ONLINE:
            docs = self.retrieve(user_query, top_k=top_k)
            context = self.format_context(docs)

            prompt_type = (
                "standard"
                if self.use_vector_db and context
                else "free"
            )

            answer = self.response_generator.generate(
                prompt_type=prompt_type,
                query=user_query,
                context=context if prompt_type == "standard" else "",
            )

            return {
                "answer": answer,
                "context": context,
                "documents": docs,
                "mode": "online",
            }

        raise ValueError(f"Unsupported RAG mode: {mode}")


# =========================================================
# DEV ENTRY (OPTIONAL)
# =========================================================
def main():
    rag = RAGSystem(use_vector_db=False)

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        result = rag.query(q, mode=RAGMode.ONLINE)
        print("Assistant:", result["answer"])


if __name__ == "__main__":
    main()
