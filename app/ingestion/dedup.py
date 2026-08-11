import hashlib
import uuid
import logging
from typing import List, Set
from app.ingestion.chunker import Chunk

logger = logging.getLogger("agentic_rag.ingestion.dedup")

import os

def generate_chunk_id(text: str, file_path: str) -> str:
    """
    Generates a deterministic UUID based on the SHA-256 hash of the chunk text
    and the file path. This guarantees idempotency: re-ingesting the same file
    produces identical IDs.
    """
    # Normalize path to prevent Windows case-sensitivity or slash differences from causing hash mismatches
    normalized_path = os.path.abspath(file_path).replace("\\", "/").lower()
    
    hasher = hashlib.sha256()
    hasher.update(normalized_path.encode("utf-8"))
    hasher.update(text.encode("utf-8"))
    hash_hex = hasher.hexdigest()
    
    # Generate a deterministic UUID version 5 using the DNS namespace and the SHA-256 hex string
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_hex))

def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Deduplicates a list of chunks within the same batch.
    Assigns the deterministic UUID to each chunk.
    """
    seen_ids: Set[str] = set()
    unique_chunks: List[Chunk] = []

    for chunk in chunks:
        file_path = chunk.metadata.get("file_path", "")
        # Compute deterministic ID
        chunk_id = generate_chunk_id(chunk.text, file_path)
        chunk.id = chunk_id

        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)
        else:
            logger.info(f"Duplicate chunk skipped in current batch: {chunk_id} ({chunk.metadata.get('filename')})")

    return unique_chunks
