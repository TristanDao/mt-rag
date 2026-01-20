"""
Reranker Module
Sử dụng Cross-Encoder để scoring lại kết quả retrieval.
"""

import os
import torch
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import config

def get_device(device: Optional[str] = None) -> str:
    """Tự động detect device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Reranker:
    """
    Reranker sử dụng cross-encoder để rerank kết quả retrieval
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Args:
            model_name: Tên cross-encoder model
            device: "cuda" hoặc "cpu" (None = auto-detect)
            batch_size: Kích thước batch (None = auto-detect)
        """
        self.device = get_device(device)
        self.model_name = config.RERANK_CONFIG.get("model_name", model_name)
        self.batch_size = config.RERANK_CONFIG.get("batch_size", batch_size)
        
        print(f"[Rerank] Initializing Reranker: {self.model_name} ({self.device})")
        
        # Check local path or download
        local_model_path = os.path.join("models", os.path.basename(self.model_name))
        
        try:
            if os.path.exists(local_model_path):
                print(f"[Rerank] Loading model from local: {local_model_path}")
                self.model = CrossEncoder(local_model_path, device=self.device)
            else:
                print(f"[Rerank] Downloading model from Hugging Face: {self.model_name}")
                self.model = CrossEncoder(self.model_name, device=self.device)
                
                # Save to local
                os.makedirs("models", exist_ok=True)
                self.model.save(local_model_path)
                print(f"[Rerank] Model saved to: {local_model_path}")
                
            print("✓ Reranker loaded successfully")
        except Exception as e:
            print(f"❌ Error loading reranker: {e}")
            self.model = None

    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents dựa trên query
        
        Args:
            query: Query text
            documents: List documents với format:
                [
                    {"document_id": "...", "score": 0.95, "text": "..."},
                    ...
                ]
            top_k: Số lượng documents sau rerank (None = giữ tất cả)
            
        Returns:
            List documents đã được rerank
        """
        if not self.model or not documents:
            return documents
        
        # Prepare pairs
        pairs = []
        doc_indices = []
        
        for i, doc in enumerate(documents):
            # Text is required for CrossEncoder
            # In Qdrant results, text is in 'payload' -> 'text' or 'combine_text'
            # Need to ensure caller passes documents with text content
            text = doc.get("text") or doc.get("content") or doc.get("payload", {}).get("text", "")
            
            if not text:
                 # If no text available, cannot rerank effectively using content
                 # Fallback to title or skip? 
                 # Let's try use 'title'
                 text = doc.get("title") or doc.get("payload", {}).get("title", "")
                 
            if text:
                pairs.append([query, text])
                doc_indices.append(i)
        
        if not pairs:
            # Cannot rerank if no text content found
            print("⚠️ Warning: No text content found for reranking. Returning original order.")
            return documents[:top_k] if top_k else documents

        # Predict scores
        # CrossEncoder returns array of scores (logits or probabilities)
        rerank_scores = self.model.predict(pairs, batch_size=self.batch_size, device=self.device)
        
        # Update scores in documents
        # Note: We only update documents that had text and were included in pairs
        reranked_docs = []
        for idx, score in zip(doc_indices, rerank_scores):
            doc = documents[idx]
            doc["rerank_score"] = float(score)
            
            # Use rerank_score as the main score for sorting
            # (Cross-encoder score is usually more accurate than bi-encoder cosine sim)
            doc["final_score"] = float(score) 
            reranked_docs.append(doc)
            
        # Sort by final_score descending
        reranked_docs.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return top_k
        if top_k:
            return reranked_docs[:top_k]
            
        return reranked_docs
