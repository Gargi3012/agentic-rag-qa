import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    PointStruct,
    SparseVector
)
from app.ingestion.chunker import Chunk
from app.retrieval.dense_embed import generate_dense_embeddings
from app.retrieval.sparse_embed import generate_sparse_embeddings

logger = logging.getLogger("agentic_rag.retrieval.qdrant")

class QdrantStore:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "").strip()
        self.host = os.getenv("QDRANT_HOST", "localhost").strip()
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = os.getenv("QDRANT_API_KEY", "").strip()
        self.collection_name = "agentic_rag_docs"
        
        # Determine connection coordinates
        if self.url:
            connection_info = self.url
        else:
            connection_info = f"{self.host}:{self.port}"
            
        logger.info(f"Connecting to Qdrant at {connection_info}...")
        try:
            # Increase timeout slightly for cloud connections
            timeout_sec = 5.0 if self.url else 2.0
            
            if self.url:
                self.client = QdrantClient(url=self.url, api_key=self.api_key or None, timeout=timeout_sec)
            else:
                self.client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key or None, timeout=timeout_sec)
                
            # Trigger request to check if server is active
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant client.")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant server at {connection_info}: {str(e)}")
            logger.warning("Falling back to local in-memory Qdrant client (data will not persist).")
            try:
                self.client = QdrantClient(":memory:")
                logger.info("Successfully initialized local in-memory Qdrant client.")
            except Exception as inner_e:
                logger.error(f"Failed to initialize in-memory Qdrant client: {str(inner_e)}")
                self.client = None

    def is_healthy(self) -> bool:
        """
        Check connectivity to Qdrant.
        """
        if not self.client:
            return False
        try:
            # Check cluster status or perform a simple request
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed for Qdrant: {str(e)}")
            return False

    def init_collection(self, collection_name: Optional[str] = None):
        """
        Initializes the Qdrant collection with dense and sparse vector configurations.
        """
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot setup collection.")
            return

        col_name = collection_name or self.collection_name
        
        try:
            # Check if collection already exists
            exists = self.client.collection_exists(collection_name=col_name)
            if exists:
                logger.info(f"Collection '{col_name}' already exists. Skipping initialization.")
                return

            logger.info(f"Creating collection '{col_name}' with dense + sparse configuration...")
            self.client.create_collection(
                collection_name=col_name,
                vectors_config={
                    "dense": VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            logger.info(f"Collection '{col_name}' successfully created.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise e

    def upsert_chunks(self, chunks: List[Chunk], collection_name: Optional[str] = None) -> int:
        """
        Upserts document chunks into Qdrant idempotently.
        To save costs, it first checks Qdrant to find chunks that already exist and skips them.
        """
        if not self.client:
            logger.error("Qdrant client unavailable. Cannot upsert chunks.")
            raise ConnectionError("Qdrant service is down.")

        if not chunks:
            return 0

        col_name = collection_name or self.collection_name
        self.init_collection(col_name)

        # Step 1: Idempotency Check (Check which chunk IDs already exist in Qdrant)
        chunk_ids = [c.id for c in chunks]
        try:
            existing_points = self.client.retrieve(
                collection_name=col_name,
                ids=chunk_ids,
                with_payload=False,
                with_vectors=False
            )
            existing_ids = {p.id for p in existing_points}
        except Exception as e:
            logger.warning(f"Failed to retrieve existing points (may be empty collection): {str(e)}")
            existing_ids = set()

        # Filter to only get chunks that don't exist yet
        new_chunks = [c for c in chunks if c.id not in existing_ids]
        
        if not new_chunks:
            logger.info("All chunks already exist in Qdrant. Skipped embedding and upsert.")
            return 0

        logger.info(f"Generating embeddings for {len(new_chunks)} new chunks (skipped {len(chunks) - len(new_chunks)})...")

        # Step 2: Batch generate dense & sparse embeddings for new chunks
        texts = [c.text for c in new_chunks]
        
        try:
            dense_vectors = generate_dense_embeddings(texts)
            sparse_vectors = generate_sparse_embeddings(texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise e

        # Step 3: Construct points and upsert
        points = []
        for idx, chunk in enumerate(new_chunks):
            points.append(PointStruct(
                id=chunk.id,
                vector={
                    "dense": dense_vectors[idx],
                    "sparse": sparse_vectors[idx]
                },
                payload={
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                }
            ))

        try:
            # Batch upsert points
            self.client.upsert(
                collection_name=col_name,
                points=points,
                wait=True
            )
            logger.info(f"Successfully upserted {len(new_chunks)} new chunks to Qdrant.")
            return len(new_chunks)
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {str(e)}")
            raise e

    def _build_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        """
        Converts a key-value dictionary to Qdrant field conditions under payload 'metadata'.
        """
        if not filter_dict:
            return None
        
        must_conditions = []
        for key, val in filter_dict.items():
            must_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=val)
                )
            )
        return models.Filter(must=must_conditions)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (Dense + Sparse) fused with Reciprocal Rank Fusion (RRF) at Qdrant level.
        Falls back to dense-only if one engine fails, and logs degraded mode.
        """
        col_name = collection_name or self.collection_name

        if not self.client:
            logger.error("Qdrant client unavailable. Cannot search.")
            raise ConnectionError("Qdrant service is down.")

        query_filter = self._build_filter(filter_dict)

        # Generate vectors
        try:
            dense_vector = generate_dense_embeddings([query])[0]
        except Exception as e:
            logger.error(f"Failed to generate query dense embedding: {str(e)}")
            raise e

        # Sparse embedding (fastembed). Fallback to dense-only if sparse fails.
        sparse_vector = None
        try:
            sparse_vector = generate_sparse_embeddings([query])[0]
        except Exception as e:
            logger.warning(f"Failed to generate query sparse embedding. Falling back to dense-only retrieval: {str(e)}")

        # Execute hybrid search with RRF
        try:
            if sparse_vector is not None:
                # Query with prefetch for dense and sparse, fused via RRF
                search_result = self.client.query_points(
                    collection_name=col_name,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=20
                        ),
                        models.Prefetch(
                            query=sparse_vector,
                            using="sparse",
                            limit=20
                        )
                    ],
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True
                )
                points = search_result.points
            else:
                # Fallback to dense-only
                logger.warning("Running degraded mode: dense-only retrieval.")
                points = self.client.search(
                    collection_name=col_name,
                    query_vector=("dense", dense_vector),
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True
                )
            
            # Map Qdrant points to structured dict outputs
            results = []
            for point in points:
                results.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "token_count": point.payload.get("token_count", 0),
                    "score": point.score
                })
            return results

        except Exception as e:
            logger.error(f"Failed to execute hybrid search in Qdrant: {str(e)}")
            # Try last-ditch fallback: dense-only via search if client works
            try:
                logger.warning("Attempting last-ditch dense-only search fallback.")
                points = self.client.search(
                    collection_name=col_name,
                    query_vector=("dense", dense_vector),
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True
                )
                return [{
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "token_count": point.payload.get("token_count", 0),
                    "score": point.score
                } for point in points]
            except Exception as inner_e:
                logger.error(f"Last-ditch search fallback failed: {str(inner_e)}")
                raise inner_e
