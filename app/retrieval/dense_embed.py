import logging
import gc
from typing import List
import torch
from sentence_transformers import SentenceTransformer

# Enforce single-thread CPU allocation to prevent Docker OOM on constrained cloud instances (512MB RAM)
torch.set_num_threads(1)

logger = logging.getLogger("agentic_rag.retrieval.dense_embed")

_model_instance = None

def get_dense_model() -> SentenceTransformer:
    """
    Returns a cached singleton instance of the dense embedding model.
    """
    global _model_instance
    if _model_instance is None:
        model_name = "all-MiniLM-L6-v2"
        logger.info(f"Loading local dense embedding model: {model_name}...")
        try:
            _model_instance = SentenceTransformer(model_name)
            logger.info("Dense embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dense embedding model: {str(e)}")
            raise e
    return _model_instance

def generate_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates dense vector representations (size 384) for a list of strings using small batch size.
    """
    if not texts:
        return []
    try:
        model = get_dense_model()
        embeddings = model.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=False)
        result = [emb.tolist() for emb in embeddings]
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Error generating dense embeddings: {str(e)}")
        raise e
