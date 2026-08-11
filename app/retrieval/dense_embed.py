import logging
from typing import List
from sentence_transformers import SentenceTransformer

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
    Generates dense vector representations (size 384) for a list of strings.
    """
    if not texts:
        return []
    try:
        model = get_dense_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.error(f"Error generating dense embeddings: {str(e)}")
        raise e
