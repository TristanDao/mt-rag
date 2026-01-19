"""
Script tạo file input JSONL cho retrieval evaluation
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Products_RAG_main.embedding import EmbeddingModel, SparseEmbeddingModel
from Products_RAG_main.vector_db import VectorDatabase
from Products_RAG_main.rerank import Reranker
import Products_RAG_main.config as config

# ============================================================================
# Global Resources
# ============================================================================
_vector_db = None
_embedding_model = None
_sparse_model = None
_reranker = None

def init_resources():
    global _vector_db, _embedding_model, _sparse_model, _reranker
    if _vector_db is None:
        print("Initializing Vector Database...")
        _vector_db = VectorDatabase(db_type=config.VECTOR_DB_CONFIG["type"])
    
    if _embedding_model is None:
        print("Initializing Embedding Model...")
        _embedding_model = EmbeddingModel(provider=config.EMBEDDING_MODEL_CONFIG.get("provider", "huggingface"))

    if _sparse_model is None and config.VECTOR_DB_CONFIG.get("sparse", False):
        print("Initializing Sparse Model...")
        sparse_model_name = config.SPARSE_CONFIG.get("model", "Qdrant/bm25")
        _sparse_model = SparseEmbeddingModel(provider="fastembed", model_name=sparse_model_name)

    if _reranker is None and config.RAG_CONFIG.get("use_reranker", False):
        print("Initializing Reranker...")
        _reranker = Reranker(
            model_name=config.RERANK_CONFIG.get("model_name"),
            device=config.RERANK_CONFIG.get("device")
        )

def retrieve(query_text: str, collection_name: str) -> List[Dict[str, Any]]:
    # Configuration
    use_rerank = config.RAG_CONFIG.get("use_reranker", False)
    
    # Get params from config (user priority)
    # search_k: số lượng doc lấy từ DB
    search_k = config.RAG_CONFIG.get("search_k", 20)
    # top_k: số lượng doc cuối cùng trả về (ghi đè tham số hàm nếu muốn config control hoàn toàn)
    final_top_k = config.RAG_CONFIG.get("top_k", 5)

    
    # 1. Vector Search
    query_vector = _embedding_model.encode_query(query_text)
    
    sparse_vector = None
    if _sparse_model:
        sparse_vector = _sparse_model.encode_single(query_text)
    
    # We need to search with payload to get text for reranker
    results = _vector_db.query(
        collection_name=collection_name,
        embedding_vector=query_vector,
        sparse_vector=sparse_vector,
        top_k=search_k
    )
    
    # 2. Rerank (Optional)
    if use_rerank and _reranker:
        results = _reranker.rerank(query_text, results, top_k=final_top_k)
        
    # 3. Format results for evaluation
    formatted_results = []
    for r in results:
        # Try to get original doc_id from payload, fallback to vector id if missing
        original_doc_id = r.get("payload", {}).get("doc_id") or r.get("id")
        
        # Use final_score if available (from reranker), otherwise standard score
        score = r.get("final_score", r.get("score"))
        
        formatted_results.append({
            "document_id": str(original_doc_id),
            "score": float(score)
        })
        
    return formatted_results

# ============================================================================
# Main Processing
# ============================================================================

def create_retrieval_input(
    queries_file: str,
    output_file: str,
    collection_key: str = "clapnq",
    top_k: int = 5
):
    # Get full collection name from config
    # Map raw key (e.g., 'clapnq') to full name in config
    collection_info = config.COLLECTION_MAPPING.get(collection_key)
    if not collection_info:
        # Try to find if key matches any name
        found = False
        for k, v in config.COLLECTION_MAPPING.items():
            if k == collection_key:
                collection_name = v["name"]
                found = True
                break
        if not found:
             # Fallback assuming user passed full name or key exists
             collection_name = collection_key
    else:
        collection_name = collection_info["name"]

    print(f"Target Collection: {collection_name}")
    
    # Initialize resources
    init_resources()
    
    # Read queries
    queries = []
    print(f"Reading queries from: {queries_file}")
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Support multiple query formats
                query_id = item.get('_id') or item.get('query_id') or item.get('id') or item.get('task_id')
                query_text = item.get('text') or item.get('query') or item.get('input')
                
                # Handle extended input format (list of dicts)
                if isinstance(query_text, list):
                     # Likely conversation history, take last user turn
                     for turn in reversed(query_text):
                         if turn.get("role") == "user" or turn.get("speaker") == "user":
                             query_text = turn.get("text")
                             break
                
                if query_id and query_text:
                    queries.append((query_id, query_text))
    except Exception as e:
        print(f"Error reading queries file: {e}")
        return

    print(f"Found {len(queries)} queries.")
    
    # Process
    print(f"Processing and writing to: {output_file}")
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries:
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{len(queries)}")
            
            contexts = retrieve(query_text, collection_name)
            
            json_obj = {
                "task_id": query_id,
                "Collection": collection_name,
                "contexts": contexts,
                "task_type": "rag",
                "input": [{"speaker": "user", "text": query_text}]
            }
            
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            
    print(f"✅ Done! Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--collection", type=str, default="clapnq")
    
    args = parser.parse_args()
    
    create_retrieval_input(
        queries_file=args.queries_file, 
        output_file=args.output_file, 
        collection_key=args.collection, 
    )


