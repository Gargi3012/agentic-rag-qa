import logging
import math
import gc
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger("agentic_rag.retrieval.reranker")

_reranker_instance = None

def get_reranker() -> CrossEncoder:
    """
    Returns a cached singleton instance of the local cross-encoder model.
    """
    global _reranker_instance
    if _reranker_instance is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading local cross-encoder: {model_name}...")
        try:
            # Loads the model locally. Will download on first run and cache.
            _reranker_instance = CrossEncoder(model_name)
            logger.info("Cross-encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {str(e)}")
            raise e
    return _reranker_instance

def sigmoid(x: float) -> float:
    """
    Sigmoid function to normalize raw Cross-Encoder log-odds scores into [0.0, 1.0].
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-ranks a list of retrieved chunks against the query using a local Cross-Encoder.
    Sorts them descending by the sigmoid-normalized score and slices to top_k.
    """
    if not chunks:
        return []
    
    try:
        reranker = get_reranker()
        
        # Prepare pairs: [query, document_text]
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Compute raw log-odds scores in small batches to save memory
        raw_scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        gc.collect()
        
        # Apply sigmoid normalization and store scores
        for idx, score in enumerate(raw_scores):
            normalized_score = sigmoid(float(score))
            chunks[idx]["rerank_score"] = normalized_score
            logger.debug(f"Chunk {chunks[idx]['id']} raw score: {score:.4f} | normalized: {normalized_score:.4f}")

        # Sort descending by the re-ranked score
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Slice to the top-k requested items
        sliced_results = reranked[:top_k]
        logger.info(f"Reranking completed. Best normalized score: {sliced_results[0]['rerank_score']:.4f}")
        return sliced_results
        
    except Exception as e:
        logger.error(f"Failed to execute cross-encoder reranking: {str(e)}. Falling back to original order.")
        # Fallback: assign original score as rerank_score
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("score", 0.0)
        return chunks[:top_k]
