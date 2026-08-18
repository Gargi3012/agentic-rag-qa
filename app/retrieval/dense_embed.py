import logging
import gc
from typing import List
from fastembed import TextEmbedding

logger = logging.getLogger("agentic_rag.retrieval.dense_embed")

_model_instance = None

def get_dense_model() -> TextEmbedding:
    """
    Returns a cached singleton instance of the dense embedding model using fastembed.
    """
    global _model_instance
    if _model_instance is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading local dense embedding model: {model_name}...")
        try:
            _model_instance = TextEmbedding(model_name=model_name)
            logger.info("Dense embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dense embedding model: {str(e)}")
            raise e
    return _model_instance

def generate_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates dense vector representations (size 384) for a list of strings using fastembed.
    """
    if not texts:
        return []
    try:
        model = get_dense_model()
        embeddings = list(model.embed(texts))
        result = [emb.tolist() for emb in embeddings]
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Error generating dense embeddings: {str(e)}")
        raise e
