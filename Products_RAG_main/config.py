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
        "name": "mt-rag-clapnq-elser-512-100-20240503",
        "corpus_file": CORPORA_DIR / "clapnq.jsonl"
    },
    "govt": {
        "name": "mt-rag-govt-elser-512-100-20240611",
        "corpus_file": CORPORA_DIR / "govt.jsonl"
    },
    "fiqa": {
        "name": "mt-rag-fiqa-beir-elser-512-100-20240501",
        "corpus_file": CORPORA_DIR / "fiqa.jsonl"
    },
    "cloud": {
        "name": "mt-rag-ibmcloud-elser-512-100-20240502",
        "corpus_file": CORPORA_DIR / "cloud.jsonl"
    }
}

# Embedding Model Settings
# Tự động detect device (CUDA nếu có, nếu không dùng CPU)
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL_CONFIG = {
    "provider": "huggingface",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Thay đổi model của bạn
    "device": AUTO_DEVICE,  # Tự động: "cuda" nếu có GPU, "cpu" nếu không
    "batch_size": 32,
    "max_length": 512
}

# Vector Database Settings
VECTOR_DB_CONFIG = {
    "type": "qdrant",  # "faiss", "chroma", "pinecone", "weaviate", etc.
    "dimension": 384,  # Phải khớp với embedding dimension
    "index_type": "flat",  # "flat", "ivf", "hnsw" cho FAISS
    # "flat": Chính xác 100%, chậm với >100K docs (dùng cho <100K)
    # "ivf": Nhanh hơn, ~95-99% accuracy (khuyên dùng cho >100K docs, như ClapNQ)
    # "hnsw": Rất nhanh, ~90-98% accuracy (nhanh nhất nhưng trade-off accuracy)
    "metric": "cosine"  # "cosine", "l2", "ip"
}

# RAG System Settings
RAG_CONFIG = {
    "top_k": 5,                 # Số lượng kết quả cuối cùng lấy ra
    "score_threshold": 0.0,
    "use_reranker": True,       # Bật reranker
}

# Reranker Settings
RERANK_CONFIG = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,                 # Số lượng sau khi rerank (thường = RAG_CONFIG["top_k"])
    "rerank_top_k": 10,         # Số lượng ứng viên lấy từ Vector DB để rerank (càng cao càng chính xác nhưng chậm)
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

