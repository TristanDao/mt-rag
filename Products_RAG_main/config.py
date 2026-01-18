"""
Configuration file cho RAG System
Điều chỉnh các tham số tại đây
"""

import os
import torch
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent / "mt-rag-benchmark-main"
CORPORA_DIR = BASE_DIR / "corpora" / "passage_level"
VECTOR_DB_DIR = BASE_DIR / "rag_system" / "vector_dbs"

# Collection mapping (từ BTC)
COLLECTION_MAPPING = {
    "clapnq": {
        "name": "clapnq",
        "corpus_file": CORPORA_DIR / "clapnq.jsonl"
    },
    "govt": {
        "name": "govt",
        "corpus_file": CORPORA_DIR / "govt.jsonl"
    },
    "fiqa": {
        "name": "fiqa",
        "corpus_file": CORPORA_DIR / "fiqa.jsonl"
    },
    "cloud": {
        "name": "ibmcloud",
        "corpus_file": CORPORA_DIR / "cloud.jsonl"
    }
}

# Embedding Model Settings
# Tự động detect device (CUDA nếu có, nếu không dùng CPU)
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL_CONFIG = {
    "provider": "huggingface",
    "model_name": "intfloat/multilingual-e5-large-instruct",  # Thay đổi model của bạn
    "device": AUTO_DEVICE,  # Tự động: "cuda" nếu có GPU, "cpu" nếu không
    "batch_size": 32,
    "max_length": 512
}

# Vector Database Settings
VECTOR_DB_CONFIG = {
    "type": "qdrant",  # "faiss", "chroma", "pinecone", "weaviate", etc.
    "dimension": 1024,  # Phải khớp với embedding dimension
    "sparse": True,
    # "index_type": "flat",  # "flat", "ivf", "hnsw" cho FAISS
    # "flat": Chính xác 100%, chậm với >100K docs (dùng cho <100K)
    # "ivf": Nhanh hơn, ~95-99% accuracy (khuyên dùng cho >100K docs, như ClapNQ)
    # "hnsw": Rất nhanh, ~90-98% accuracy (nhanh nhất nhưng trade-off accuracy)
    "metric": "cosine"  # "cosine", "l2", "ip"
}

SPARSE_CONFIG = {
    "type" : "bm25",
    "model" : "Qdrant/bm25"
}


# RAG System Settings
RAG_CONFIG = {
    "search_k": 30,             # Số lượng documents tìm kiếm từ Vector DB
    "top_k": 10,                 # Số lượng kết quả cuối cùng trả về (top K final)
    "score_threshold": 0.0,
    "use_reranker": False,       # Bật reranker
}

# Reranker Settings
RERANK_CONFIG = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "batch_size": 32,
    "device": AUTO_DEVICE
}

# LLM Settings (cho generation task sau này)
LLM_CONFIG = {
    "provider": "openai",  # "openai", "huggingface", "azure", etc.
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 512
}