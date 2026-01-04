# ============================================================
#   VECTOR DATABASE LAYER — CLEAN VERSION (NO SSL ERRORS)
# ============================================================

# from pymongo import MongoClient
# import certifi
# from chromadb import HttpClient
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams, SparseVectorParams, SparseIndexParams, SparseVector, NamedSparseVector
# from supabase import create_client

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

            try:
                from pymongo import MongoClient
                import certifi
            except ImportError:
                raise ImportError("pymongo is not installed. Please install it to use MongoDB.")

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

            try:
                from chromadb import HttpClient
            except ImportError:
                raise ImportError("chromadb is not installed. Please install it to use ChromaDB.")

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

            try:
                from supabase import create_client
            except ImportError:
                raise ImportError("supabase is not installed. Please install it to use Supabase.")

            self.client = create_client(url, key)

        else:
            raise ValueError(f"Unsupported DB: {db_type}")

    # ================================================================
    #   CREATE COLLECTION IF NOT EXISTS
    # ================================================================
    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 384):
        if self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name):
                print(f"[VectorDB] Creating Qdrant collection: {collection_name} (size={vector_size})")
                
                vectors_config = {"default": VectorParams(size=vector_size, distance=Distance.COSINE)}
                sparse_vectors_config = None
                
                sparse_vectors_config = {
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
        
        elif self.db_type == "chromadb":
            self.client.get_or_create_collection(collection_name)


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
            points = []
            for doc in data:
                # doc["embedding"] is dense vector (list[float])
                # check if doc has "sparse_embedding" -> dict {"indices": [], "values": []}
                
                vector_dict = {
                    "default": doc["embedding"]
                }
                
                if "sparse_embedding" in doc:
                    vector_dict["bm25"] = models.SparseVector(
                        indices=doc["sparse_embedding"]["indices"],
                        values=doc["sparse_embedding"]["values"]
                    )
                
                points.append(PointStruct(
                    id=doc["id"],
                    vector=vector_dict,
                    payload=doc.get("metadata", {})
                ))
            self.client.upsert(collection_name=collection_name, points=points)
            return

        # ----------------- Supabase -----------------
        if self.db_type == "supabase":
            self.client.table(collection_name).insert(data).execute()
            return

    # ================================================================
    #   VECTOR SEARCH
    # ================================================================
    def query(self, collection_name: str, embedding_vector: List[float], sparse_vector: Dict[str, Any] = None, top_k: int = 5):
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
            # Hybrid search if sparse_vector is provided
            if sparse_vector:
                # IF we want proper hybrid (CombSUM / RRF):
                # We need 2 requests and merge.
                # But to keep it simple and since I cannot guarantee Qdrant version:
                # We will perform 2 searches if sparse provided and fuse them in Python?
                # User asked for "Using Qdrant defaults" which might imply just having them available.
                # Let's try to use the 'search_batch' or just 2 calls.
                
                # ACTUALLY, usually "Hybrid Search" implies Reciprocal Rank Fusion of Retrieval.
                # Let's do RRF manually here for clarity and Robustness.
                
                limit_mult = 1 # fetch more to fuse
                
                dense_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=models.NamedVector(
                        name="default",
                        vector=embedding_vector
                    ),
                    limit=top_k * limit_mult,
                    with_payload=True
                )
                
                sparse_struct = models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
                
                sparse_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=models.NamedSparseVector(
                        name="bm25",
                        vector=sparse_struct
                    ),
                    limit=top_k * limit_mult,
                    with_payload=True
                )
                
                # RRF FUSION
                def rrf_score(rank, k=60):
                    return 1 / (k + rank)
                
                doc_scores = {}
                doc_payloads = {}
                
                # Process dense
                for rank, hit in enumerate(dense_hits):
                    doc_scores[hit.id] = doc_scores.get(hit.id, 0) + rrf_score(rank + 1)
                    doc_payloads[hit.id] = hit.payload
                    
                # Process sparse
                for rank, hit in enumerate(sparse_hits):
                    doc_scores[hit.id] = doc_scores.get(hit.id, 0) + rrf_score(rank + 1)
                    if hit.id not in doc_payloads: doc_payloads[hit.id] = hit.payload

                # Sort
                sorted_ids = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
                
                # Format output
                final_output = []
                for valid_id, score in sorted_ids:
                    # Construct pseudo-object to match interface
                    # We lost original 'score' (cosine/dot) in favor of RRF score
                    final_output.append(models.ScoredPoint(
                        id=valid_id,
                        version=0,
                        score=score,
                        payload=doc_payloads[valid_id],
                        vector=None
                    ))
                
                results = final_output
                
            else:
                # Standard Dense Search (Backward compatible if data was inserted as unnamed vector??)
                # If we changed insertion to use 'default' name, we MUST query with 'default' name
                # OR Qdrant allows querying unnamed if only 1 vector? 
                # Ideally, if we use named vectors, we must always use named query.
                
                # NOTE: If the collection was created with 'vectors_config={"default": ...}', 
                # we MUST use named vector "default". 
                # If it was created with 'vectors_config=VectorParams(...)', we use unnamed.
                # Since we updated 'create_collection' to use named 'default', we should try named first.
                
                try:
                    results = self.client.search(
                        collection_name=collection_name,
                        query_vector=models.NamedVector(
                            name="default",
                            vector=embedding_vector
                        ),
                        limit=top_k,
                        with_payload=True
                    )
                except Exception:
                    # Fallback to unnamed if collection was old?
                    # But user is "re-inserting", so we assume new schema.
                    # But just in case:
                    results = self.client.search(
                        collection_name=collection_name,
                        query_vector=embedding_vector,
                        limit=top_k,
                        with_payload=True
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

