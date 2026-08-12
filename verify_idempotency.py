import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.config import Config
from app.ingestion import load_directory, chunk_document, deduplicate_chunks
from app.retrieval.qdrant_client import QdrantStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag.verify_idempotency")

def verify_idempotency():
    logger.info("Initializing Qdrant Store client...")
    store = QdrantStore()
    
    # We will use a special check collection
    check_collection = "agentic_rag_idempotency_check"
    
    # Clean check collection
    try:
        if store.client.collection_exists(collection_name=check_collection):
            logger.info(f"Deleting existing check collection '{check_collection}'...")
            store.client.delete_collection(collection_name=check_collection)
    except Exception as e:
        logger.warning(f"Failed to check/delete collection: {str(e)}")
    
    logger.info(f"Initializing fresh check collection '{check_collection}'...")
    store.init_collection(check_collection)
    
    data_dir = os.path.abspath("data")
    logger.info(f"Loading documents from {data_dir}...")
    docs = load_directory(data_dir)
    logger.info(f"Loaded {len(docs)} documents.")
    
    # Chunking
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        all_chunks.extend(chunks)
    logger.info(f"Generated {len(all_chunks)} raw splits.")
    
    # Deduplicate in-batch
    unique_chunks = deduplicate_chunks(all_chunks)
    logger.info(f"Deduplicated to {len(unique_chunks)} unique chunks.")
    
    # Run 1
    logger.info("\n--- STARTING INGESTION RUN 1 ---")
    orig_name = store.collection_name
    store.collection_name = check_collection
    
    upserted_1 = store.upsert_chunks(unique_chunks)
    count_1 = store.client.count(collection_name=check_collection).count
    logger.info(f"RUN 1 COMPLETE.")
    logger.info(f"-> Chunks upserted: {upserted_1}")
    logger.info(f"-> Qdrant vector count: {count_1}")
    
    # Run 2
    logger.info("\n--- STARTING INGESTION RUN 2 (IDEMPOTENCY CHECK) ---")
    upserted_2 = store.upsert_chunks(unique_chunks)
    count_2 = store.client.count(collection_name=check_collection).count
    logger.info(f"RUN 2 COMPLETE.")
    logger.info(f"-> Chunks upserted: {upserted_2}")
    logger.info(f"-> Qdrant vector count: {count_2}")
    
    # Assertions
    assert count_1 == count_2, f"FAILED: Count changed from {count_1} to {count_2}!"
    assert upserted_2 == 0, f"FAILED: Upserted {upserted_2} chunks on second run, should be 0!"
    
    logger.info("\n🎉 SUCCESS: Ingestion is 100% idempotent!")
    logger.info(f"Vector count before second run: {count_1}")
    logger.info(f"Vector count after second run: {count_2}")
    logger.info(f"Total embedding api calls saved on second run: {len(unique_chunks)}")
    
    # Restore collection name
    store.collection_name = orig_name

if __name__ == "__main__":
    verify_idempotency()
