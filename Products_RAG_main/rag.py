from embedding import EmbeddingModel
from vector_db import VectorDatabase
from semetic_router.samples import chitchatSample, productsSample
from semetic_router.route import Route, SemanticRouter
from reflection import Reflection

from openai import OpenAI
# ⚠ KHÔNG import google.genai để tránh lỗi evaluator
# from google.genai import Client as GeminiClient

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()


from Products_RAG_main.rerank import Reranker
from Products_RAG_main import config

class RAGSystem:
    def __init__(
            self,
            db_type: str = "mongodb",
            embedding_provider: str = "huggingface",
            llm_provider: str = "openai",        # ⚠ đặt mặc định OPENAI
            embedding_model: Optional[str] = None,
            llm_model: Optional[str] = None,
            collection_name: str = "products",
            routes: Optional[List[Route]] = None
    ):
        self.collection_name = collection_name
        
        # Load settings from config
        self.rerank_config = config.RAG_CONFIG
        
        # -------------------------------
        # Init Database
        # -------------------------------
        print(f"[RAG] Initializing {db_type} database...")
        self.vector_db = VectorDatabase(db_type=db_type)
        print("[RAG] Database OK!")

        # -------------------------------
        # Init Embedding Model
        # -------------------------------
        print(f"[RAG] Initializing embedding model: {embedding_provider}")
        self.embedding_model = EmbeddingModel(
            provider=embedding_provider,
            model_name=embedding_model
        )
        print("[RAG] Embedding OK!")

        # -------------------------------
        # Init Reranker (Optional)
        # -------------------------------
        self.reranker = None
        if self.rerank_config.get("use_reranker", False):
            try:
                print("[RAG] Initializing Reranker...")
                self.reranker = Reranker(
                    model_name=config.RERANK_CONFIG.get("model_name"),
                    device=config.RERANK_CONFIG.get("device")
                )
                print("[RAG] Reranker OK!")
            except Exception as e:
                print(f"[RAG] ⚠️ Failed to load Reranker: {e}")

        # -------------------------------
        # Init LLM (OpenAI only for evaluator)
        # -------------------------------
        print(f"[RAG] Initializing LLM provider: {llm_provider}")
        self.llm_provider = llm_provider.lower()

        if self.llm_provider == "openai":
            self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_model = llm_model or "gpt-4o-mini"
            print("[RAG] LLM OpenAI OK!\n")

        elif self.llm_provider == "gemini":
            raise RuntimeError(
                "Gemini is not supported in evaluator mode. "
                "Please switch llm_provider='openai'."
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        # -------------------------------
        # Semantic Router + Reflection
        # -------------------------------
        print("[RAG] Initializing router + reflection...")
        self.reflection = Reflection(self.llm_client)

        if routes:
            self.router = SemanticRouter(self.embedding_model, routes)
        else:
            self.router = None

        print("[RAG] System Ready!\n")

    # ----------------------------------------------------------------------

    def route_query(self, query: str, message: List[Dict]) -> tuple[str, str]:
        if not self.router:
            return "products", query

        rewritten_query = self.reflection.rewrite(message, query)
        print(f"[RAG] Rewritten query: {rewritten_query}")

        best_route = self.router.guide(rewritten_query)[1]
        print(f"[RAG] Semantic Route: {best_route}")

        return best_route, rewritten_query

    # ----------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Check if rerank used
        use_rerank = self.reranker is not None
        
        # Determine initial search k (candidates)
        search_k = config.RERANK_CONFIG.get("rerank_top_k", 20) if use_rerank else top_k
        
        query_embedding = self.embedding_model.encode_single(query)
        
        results = self.vector_db.query(
            collection_name=self.collection_name,
            embedding_vector=query_embedding,
            top_k=search_k
        )
        
        # Apply Reranker
        if use_rerank:
            results = self.reranker.rerank(query, results, top_k=top_k)
            
        return results

    # ----------------------------------------------------------------------

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"Tài liệu {i+1}:\n{doc['information']}\n"
            for i, doc in enumerate(results)
        )

    # ----------------------------------------------------------------------

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""
Bạn là nhân viên tư vấn điện thoại chuyên nghiệp.

Thông tin sản phẩm:
{context}

Câu hỏi khách hàng: {query}

Trả lời thân thiện, chính xác, dựa trên thông tin.
"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Bạn là nhân viên tư vấn bán hàng."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content

    # ----------------------------------------------------------------------

    def query(self, user_query: str, top_k: int = 5, messages: Optional[List[Dict]] = None):
        if messages is None:
            messages = [{"role": "system", "content": "Bạn là nhân viên tư vấn điện thoại."}]

        route, rewritten = self.route_query(user_query, messages)

        if route == "chitchat":
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages + [{"role": "user", "content": rewritten}]
            )
            return {"answer": response.choices[0].message.content}

        # Route: products
        results = self.retrieve(rewritten, top_k)
        context = self.format_context(results)
        answer = self.generate_response(rewritten, context)

        return {
            "answer": answer,
            "source": results,
            "query": rewritten,
            "context": context
        }

# --------------------------------------------------------------------------

def main():
    routes = [
        Route(name="products", samples=productsSample),
        Route(name="chitchat", samples=chitchatSample),
    ]

    rag = RAGSystem(
        db_type="mongodb",
        embedding_provider="huggingface",
        llm_provider="openai",
        collection_name="products",
        routes=routes,
    )

    print("\n=== Chat Mode ===\n")

    messages = []
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        result = rag.query(user_input, messages=messages)
        print("\nTrợ lý:", result["answer"], "\n")

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
