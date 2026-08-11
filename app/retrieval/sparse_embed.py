import logging
from typing import List, Dict, Any
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

logger = logging.getLogger("agentic_rag.retrieval.sparse_embed")

_sparse_model_instance = None

def get_sparse_model() -> SparseTextEmbedding:
    """
    Returns a cached singleton instance of the FastEmbed sparse model.
    """
    global _sparse_model_instance
    if _sparse_model_instance is None:
        model_name = "Qdrant/bm25"
        logger.info(f"Loading local sparse embedding model: {model_name}...")
        try:
            # FastEmbed downloads and caches the model locally on first instantiation
            _sparse_model_instance = SparseTextEmbedding(model_name=model_name)
            logger.info("Sparse embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sparse embedding model: {str(e)}")
            raise e
    return _sparse_model_instance

def generate_sparse_embeddings(texts: List[str]) -> List[SparseVector]:
    """
    Generates sparse vector representations for a list of strings.
    Converts fastembed output to Qdrant's SparseVector structure.
    """
    if not texts:
        return []
    try:
        model = get_sparse_model()
        # model.embed returns a generator yielding SparseEmbedding objects
        embeddings = list(model.embed(texts))
        
        sparse_vectors = []
        for emb in embeddings:
            # emb has .indices (list of ints) and .values (array of floats)
            sparse_vectors.append(SparseVector(
                indices=emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices),
                values=emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
            ))
        return sparse_vectors
    except Exception as e:
        logger.error(f"Error generating sparse embeddings: {str(e)}")
        raise e
