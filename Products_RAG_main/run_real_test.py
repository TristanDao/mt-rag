"""
Script thực hiện retrieval trên tập dữ liệu test thực tế (rag_taskAC.jsonl + final_rewrite.jsonl)
Output kết quả bao gồm contexts để đánh giá hoặc debug.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Products_RAG_main.embedding import EmbeddingModel, SparseEmbeddingModel
from Products_RAG_main.vector_db import VectorDatabase
from Products_RAG_main.rerank import Reranker
import Products_RAG_main.config as config

def load_rewritten_queries(file_path: str) -> Dict[str, str]:
    """
    Load rewritten queries from jsonl file.
    Returns: dict { _id: rewritten_text }
    """
    mapping = {}
    print(f"Reading rewritten queries from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return mapping
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                _id = item.get("_id")
                text = item.get("text", "")
                
                # Remove prefix if present
                if text.startswith("|user|:"):
                    text = text.replace("|user|:", "", 1).strip()
                elif text.startswith("User:"): 
                     text = text.replace("User:", "", 1).strip()
                
                if _id:
                    mapping[_id] = text.strip()
            except Exception:
                continue
    return mapping

def resolve_collection_name(collection_key: str) -> str:
    """Map collection key (e.g. 'clapnq') to full collection name from config"""
    # 1. Check direct mapping
    info = config.COLLECTION_MAPPING.get(collection_key)
    if info:
        return info["name"]
    
    # 2. Check if key is already a valid name in values
    for k, v in config.COLLECTION_MAPPING.items():
        if v["name"] == collection_key:
            return collection_key
            
    # 3. Fallback: use key as name
    return collection_key

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    rag_task_file = data_dir / "rag_taskAC.jsonl"
    rewrite_file = data_dir / "final_rewrite.jsonl"
    output_file = base_dir / "Products_RAG_main" / "real_test_output.jsonl"

    print(f"Input Task File: {rag_task_file}")
    print(f"Rewrite File: {rewrite_file}")
    print(f"Output File: {output_file}")

    # 1. Load Rewrites
    rewrite_map = load_rewritten_queries(str(rewrite_file))
    print(f"Loaded {len(rewrite_map)} rewritten queries.")

    # 2. Init Resources
    print("Initializing Models...")
    # Config keys
    retrieval_top_k = config.RAG_CONFIG.get("search_k", 20)      # Fetch from DB
    final_top_k = config.RAG_CONFIG.get("top_k", 5)             # After rerank
    
    vector_db = VectorDatabase(db_type=config.VECTOR_DB_CONFIG["type"])
    embedding_model = EmbeddingModel(provider=config.EMBEDDING_MODEL_CONFIG.get("provider", "huggingface"))
    
    # Sparse
    sparse_model = None
    if config.VECTOR_DB_CONFIG.get("sparse", False):
        sparse_model_name = config.SPARSE_CONFIG.get("model", "Qdrant/bm25")
        print(f"Initializing Sparse Model: {sparse_model_name}")
        sparse_model = SparseEmbeddingModel(provider="fastembed", model_name=sparse_model_name)
    
    # Reranker
    reranker = None
    if config.RAG_CONFIG.get("use_reranker", False):
        print("Initializing Reranker...")
        reranker = Reranker(
            model_name=config.RERANK_CONFIG.get("model_name"),
            device=config.RERANK_CONFIG.get("device")
        )

    # 3. Process
    results_list = []
    
    with open(rag_task_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} tasks...")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines):
            try:
                task = json.loads(line.strip())
            except:
                continue
                
            task_id = task.get("task_id")
            collection_key = task.get("Collection")
            
            # --- 3.1 Get Query ---
            # Try rewritten first
            query_text = rewrite_map.get(task_id)
            
            # Fallback to last user turn if no rewrite
            if not query_text:
                input_data = task.get("input", [])
                if isinstance(input_data, list):
                    for turn in reversed(input_data):
                        if turn.get("speaker") == "user":
                            query_text = turn.get("text")
                            break
            
            if not query_text:
                # Still empty? Skip
                continue

            # --- 3.2 Resolve Collection ---
            collection_name = resolve_collection_name(collection_key)
            if not collection_name:
                continue

            # --- 3.3 Embed ---
            query_vec = embedding_model.encode_query(query_text)
            
            sparse_vec = None
            if sparse_model:
                sparse_vec = sparse_model.encode_single(query_text)

            # --- 3.4 Retrieve ---
            try:
                db_results = vector_db.query(
                    collection_name=collection_name,
                    embedding_vector=query_vec,
                    sparse_vector=sparse_vec,
                    top_k=retrieval_top_k
                )
            except Exception as e:
                print(f"Error querying {collection_name}: {e}")
                db_results = []
            
            # --- 3.5 Rerank ---
            if reranker and db_results:
                db_results = reranker.rerank(query_text, db_results, top_k=final_top_k)
            else:
                # Slice to final_top_k if no reranker
                db_results = db_results[:final_top_k]

            # --- 3.6 Format Output ---
            contexts = []
            for r in db_results:
                payload = r.get("payload", {})
                contexts.append({
                    "document_id": r.get("id"), # Point ID
                    "original_id": payload.get("doc_id"), # Original ID from corpus
                    "score": r.get("score") if not r.get("final_score") else r.get("final_score"), # Handle rerank score
                    "text": payload.get("text", "")
                })
            
            # Prepare output object: copy original task and update with results
            output_obj = task.copy()
            output_obj["query"] = query_text
            output_obj["contexts"] = contexts
            
            # Write line immediately
            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

    print(f"\nDone! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
