# ============================================================
#   VECTOR DATABASE LAYER — CLEAN VERSION (NO SSL ERRORS)
# ============================================================

from pymongo import MongoClient
import certifi
from chromadb import HttpClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from supabase import create_client

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    def __init__(self, db_type: str):
        """
        db_type: mongodb | chromadb | qdrant | supabase
        """
        self.db_type = db_type.lower()

        # ---------------------------------------------------------
        #  MONGODB (LOCAL DATABASE)
        # ---------------------------------------------------------
        if self.db_type == "mongodb":
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

            print(f"[VectorDB] Using MongoDB: {mongo_uri}")

            # ❌ KHÔNG dùng tls / ssl khi chạy local → tránh lỗi handshake
            self.client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)


        # ---------------------------------------------------------
        # CHROMADB
        # ---------------------------------------------------------
        elif self.db_type == "chromadb":
            host = os.getenv("CHROMADB_HOST", "localhost")
            port = int(os.getenv("CHROMADB_PORT", 8000))
            print(f"[VectorDB] Using ChromaDB: {host}:{port}")

            self.client = HttpClient(host=host, port=port)

        # ---------------------------------------------------------
        # QDRANT
        # ---------------------------------------------------------
        elif self.db_type == "qdrant":
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            key = os.getenv("QDRANT_KEY", None)

            print(f"[VectorDB] Using Qdrant: {url}")

            self.client = QdrantClient(url=url, api_key=key)

        # ---------------------------------------------------------
        # SUPABASE
        # ---------------------------------------------------------
        elif self.db_type == "supabase":
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if not url or not key:
                raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

            print(f"[VectorDB] Using Supabase")

            self.client = create_client(url, key)

        else:
            raise ValueError(f"Unsupported DB: {db_type}")

    # ================================================================
    #   INSERT DOCUMENTS
    # ================================================================
    def insert(self, data: List[Dict[str, Any]], collection_name: str):
        if not data:
            raise ValueError("No data to insert.")

        # ----------------- MongoDB -----------------
        if self.db_type == "mongodb":
            db = self.client["vector_db"]
            collection = db[collection_name]
            collection.insert_many(data)
            return

        # ----------------- ChromaDB -----------------
        if self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(collection_name)
            for row in data:
                collection.add(
                    ids=[row["id"]],
                    embeddings=[row["embedding"]],
                    documents=[row.get("document", "")],
                    metadatas=[row.get("metadata", {})]
                )
            return

        # ----------------- Qdrant -----------------
        if self.db_type == "qdrant":
            points = [
                PointStruct(
                    id=doc["id"],
                    vector=doc["embedding"],
                    payload=doc.get("metadata", {})
                )
                for doc in data
            ]
            self.client.upsert(collection_name=collection_name, points=points)
            return

        # ----------------- Supabase -----------------
        if self.db_type == "supabase":
            self.client.table(collection_name).insert(data).execute()
            return

    # ================================================================
    #   VECTOR SEARCH
    # ================================================================
    def query(self, collection_name: str, embedding_vector: List[float], top_k: int = 5):
        # ----------------- MongoDB -----------------
        if self.db_type == "mongodb":
            db = self.client["vector_db"]
            collection = db[collection_name]

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": embedding_vector,
                        "numCandidates": 100,
                        "limit": top_k
                    }
                },
                {"$project": {"_id": 0, "information": 1, "metadata": 1,
                              "score": {"$meta": "vectorSearchScore"}}}
            ]

            return list(collection.aggregate(pipeline))

        # ----------------- ChromaDB -----------------
        if self.db_type == "chromadb":
            col = self.client.get_or_create_collection(collection_name)
            result = col.query(query_embeddings=[embedding_vector], n_results=top_k)

            output = []
            ids = result["ids"][0]
            docs = result["documents"][0]
            metas = result["metadatas"][0]
            dists = result["distances"][0]
            for i in range(len(ids)):
                output.append({
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "score": dists[i],
                })
            return output

        # ----------------- Qdrant -----------------
        if self.db_type == "qdrant":
            results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding_vector,
                limit=top_k
            )

            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in results
            ]

        # ----------------- Supabase -----------------
        if self.db_type == "supabase":
            response = self.client.rpc("match_vectors", {
                "query_vector": embedding_vector,
                "match_count": top_k
            }).execute()

            if response.error:
                raise RuntimeError(response.error)

            return response.data

        raise ValueError("Unsupported DB type.")

