from enum import Enum
from typing import List, Dict, Any, Optional
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# PATH SETUP
# Products_RAG-main/..  →  NLP_project (project root)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# SAFE IMPORTS (FILE NẰM Ở PROJECT ROOT)
# =========================================================
from response_generator import ResponseGenerator

# Các module nội bộ trong Products_RAG-main
from reflection import Reflection
from semetic_router.route import Route, SemanticRouter
from semetic_router.samples import chitchatSample, productsSample

load_dotenv()


# =========================================================
# MODE
# =========================================================
class RAGMode(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"   # MT-RAG benchmark


# =========================================================
# RAG SYSTEM
# =========================================================
class RAGSystem:
    def __init__(
        self,
        db_type: str = "mongodb",
        embedding_provider: str = "huggingface",
        llm_provider: str = "openai",
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        collection_name: str = "products",
        routes: Optional[List[Route]] = None
    ):
        self.collection_name = collection_name

        # =================================================
        # OFFLINE / ONLINE SWITCH
        # =================================================
        self.offline = os.getenv("RAG_OFFLINE_MODE") == "1"

        if self.offline:
            print("[RAG] OFFLINE MODE: skip embedding & vector DB")
            self.embedding_model = None
            self.vector_db = None
        else:
            # ONLINE ONLY (không dùng trong MT-RAG benchmark)
            from embedding import EmbeddingModel
            from vector_db import VectorDatabase

            print(f"[RAG] Initializing {db_type} database...")
            self.vector_db = VectorDatabase(db_type=db_type)
            print("[RAG] Database OK!")

            print(f"[RAG] Initializing embedding model: {embedding_provider}")
            self.embedding_model = EmbeddingModel(
                provider=embedding_provider,
                model_name=embedding_model
            )
            print("[RAG] Embedding OK!")

        # =================================================
        # LLM
        # =================================================
        if llm_provider.lower() != "openai":
            raise ValueError("Only OpenAI provider is supported")

        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model or "gpt-4o-mini"
        print("[RAG] LLM OpenAI OK!")

        # =================================================
        # RESPONSE GENERATOR
        # =================================================
        self.response_generator = ResponseGenerator(
            llm_client=self.llm_client,
            model=self.llm_model
        )

        # =================================================
        # ROUTER + REFLECTION
        # =================================================
        self.reflection = Reflection(self.llm_client)

        if routes and not self.offline:
            self.router = SemanticRouter(self.embedding_model, routes)
        else:
            # OFFLINE: router disabled hoàn toàn
            self.router = None

        print("[RAG] System Ready!\n")

    # =====================================================
    # FORMAT CONTEXT (MT-RAG dùng key `text`)
    # =====================================================
    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"Tài liệu {i+1}:\n{doc.get('text', '')}\n"
            for i, doc in enumerate(docs)
        )

    # =====================================================
    # QUERY ENTRYPOINT
    # =====================================================
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        messages: Optional[List[Dict]] = None,
        mode: RAGMode = RAGMode.ONLINE,
        provided_contexts: Optional[List[Dict]] = None
    ):
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # -----------------------------
        # OFFLINE MODE (MT-RAG)
        # -----------------------------
        if mode == RAGMode.OFFLINE:
            if not provided_contexts:
                raise ValueError("OFFLINE mode requires provided_contexts")

            context = self.format_context(provided_contexts)

            answer = self.response_generator.generate(
                prompt_type="standard",
                query=user_query,
                context=context
            )

            return {
                "answer": answer,
                "query": user_query,
                "context": context,
                "mode": "offline"
            }

        # -----------------------------
        # ONLINE MODE (KHÔNG DÙNG TRONG BENCHMARK)
        # -----------------------------
        raise RuntimeError(
            "ONLINE mode is disabled in MT-RAG benchmark execution"
        )


# =========================================================
# DEV ENTRY (KHÔNG DÙNG TRONG BENCHMARK)
# =========================================================
def main():
    routes = [
        Route(name="products", samples=productsSample),
        Route(name="chitchat", samples=chitchatSample),
    ]

    rag = RAGSystem(routes=routes)

    messages = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        result = rag.query(user_input, messages=messages)
        print("Assistant:", result["answer"])

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
