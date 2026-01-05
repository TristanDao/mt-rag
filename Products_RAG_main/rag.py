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
# RAG SYSTEM
# =========================================================
class RAGSystem:
    def __init__(
        self,
        use_vector_db: bool = False,
        db_type: str = "qdrant",   # 👈 DEFAULT LÀ QDRANT
        embedding_provider: str = "huggingface",
        embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        collection_name: str = "mt-rag-clapnq-elser-512-100-20240503",
    ):
        self.use_vector_db = use_vector_db
        self.collection_name = collection_name

        # =================================================
        # VECTOR DB + EMBEDDING (OPTIONAL)
        # =================================================
        if self.use_vector_db:
            from Products_RAG_main.vector_db import VectorDatabase
            from Products_RAG_main.embedding import EmbeddingModel

            print("[RAG] Initializing VectorDB (Qdrant) + Embedding")

            self.vector_db = VectorDatabase(db_type=db_type)

            self.embedding_model = EmbeddingModel(
                provider=embedding_provider,
                model_name=embedding_model
            )

            self.vector_db.create_collection_if_not_exists(
                collection_name=self.collection_name,
                vector_size=self.embedding_model.vector_size
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
    def retrieve(self, query: str, top_k: int = 5):
        if not self.use_vector_db or self.vector_db is None:
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
    def format_context(self, docs):
        if not docs:
            return ""

        chunks = []
        for i, doc in enumerate(docs):
            payload = doc.get("payload", {})
            text = payload.get("text") or payload.get("content") or ""
            if text:
                chunks.append(f"Tài liệu {i+1}:\n{text}")

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
        # if mode == RAGMode.ONLINE:
        #     docs = self.retrieve(user_query, top_k) if self.use_vector_db else []
        #     context = self.format_context(docs)

        #     answer = self.response_generator.generate(
        #         prompt_type="standard" if self.use_vector_db else "free",
        #         query=user_query,
        #         context=context if self.use_vector_db else ""
        #     )

        #     return {
        #         "answer": answer,
        #         "context": context,
        #         "documents": docs,
        #         "mode": "online",
        #     }

        # =================================================
        # ONLINE MODE
        # =================================================
        if mode == RAGMode.ONLINE:
            if self.use_vector_db:
                docs = self.retrieve(user_query, top_k=top_k)
                context = self.format_context(docs)
            else:
                docs = []
                context = ""
            if self.vector_db is None:
                answer = self.response_generator.generate(
                prompt_type="free",
                query=user_query,
                context= ""
                )
            else:
                answer = self.response_generator.generate(
                    prompt_type="standard",
                    query=user_query,
                    context=context,
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
