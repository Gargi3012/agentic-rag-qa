import time
import logging
import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, Security, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from app.api.frontend_html import get_frontend_html

from app.config import Config
from app.api.auth import get_api_key
from app.api.rate_limiter import rate_limit_dependency
from app.api.logging import setup_logging
from app.ingestion import load_directory, chunk_document, deduplicate_chunks
from app.retrieval import QdrantStore
from app.generation import AgenticQueryPipeline

# Setup structured logging
setup_logging()
logger = logging.getLogger("agentic_rag.api.main")

app = FastAPI(
    title="Cost-Efficient Agentic RAG QA Service",
    description="An agent-driven RAG QA API leveraging self-hosted Qdrant, dense+sparse embeddings, and gpt-4o-mini with self-correction loops.",
    version="0.1.0",
)

# Global clients/services
store = QdrantStore()
pipeline = AgenticQueryPipeline(store=store)

# In-memory metrics tracking
class MetricsTracker:
    def __init__(self):
        self.total_queries = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.refusals = 0
        self.successes = 0
        self.history: List[Dict[str, Any]] = []
        self.lock_stats = False  # Basic concurrency helper

    def record_query(self, metrics: Dict[str, Any]):
        self.total_queries += 1
        self.total_latency_ms += metrics.get("latency_ms", 0.0)
        self.total_tokens += metrics.get("tokens_used", 0)
        self.total_cost += metrics.get("cost_estimate", 0.0)
        
        status_val = metrics.get("status", "success")
        if "refusal" in status_val or status_val == "model_refusal":
            self.refusals += 1
        else:
            self.successes += 1
            
        # Limit history to last 50 items
        self.history.append(metrics)
        if len(self.history) > 50:
            self.history.pop(0)

metrics_tracker = MetricsTracker()
start_time = time.time()

# Request/Response Schemas
class IngestRequest(BaseModel):
    directory_path: str = Field(..., description="Absolute local path to the directory containing PDF, HTML, or MD documents.")

class IngestResponse(BaseModel):
    status: str
    files_processed: List[str]
    chunks_created: int
    chunks_upserted: int
    time_taken_seconds: float

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the agentic RAG system.")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional key-value filters matching document metadata.")
    k: Optional[int] = Field(5, description="Number of chunks to retrieve.")

class ChunkDetail(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    answer: str
    status: str
    confidence: str
    retries: int
    cost: float
    latency_ms: float
    chunks: List[ChunkDetail]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime_seconds: float
    qdrant_connected: bool

class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_queries: int
    success_rate_percent: float
    average_latency_ms: float
    total_tokens_used: int
    total_cost_usd: float
    query_history: List[Dict[str, Any]]

# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Diagnostics endpoint to verify API server status, uptime, and Qdrant connectivity.
    """
    qdrant_connected = store.is_healthy()
    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        timestamp=time.time(),
        version="0.1.0",
        uptime_seconds=time.time() - start_time,
        qdrant_connected=qdrant_connected
    )

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents(
    request: IngestRequest,
    api_key: str = Security(rate_limit_dependency)
):
    """
    Ingests all supported documents (PDF, HTML, MD) from a local directory path.
    Chunks and loads them into Qdrant idempotently.
    """
    t_start = time.time()
    try:
        # Load documents
        docs = load_directory(request.directory_path)
        if not docs:
            logger.warning(f"No valid documents found in directory: {request.directory_path}")
            return IngestResponse(
                status="no_files_found",
                files_processed=[],
                chunks_created=0,
                chunks_upserted=0,
                time_taken_seconds=time.time() - t_start
            )

        # Chunk documents
        all_chunks = []
        files_processed = list(set(doc.metadata["filename"] for doc in docs))
        for doc in docs:
            chunks = chunk_document(doc, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
            all_chunks.extend(chunks)

        # Deduplicate chunks in-batch
        unique_chunks = deduplicate_chunks(all_chunks)

        # Upsert into Qdrant (checking for pre-existing IDs inside to save embedding costs)
        chunks_upserted = store.upsert_chunks(unique_chunks)

        return IngestResponse(
            status="success",
            files_processed=files_processed,
            chunks_created=len(all_chunks),
            chunks_upserted=chunks_upserted,
            time_taken_seconds=time.time() - t_start
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def run_query(
    request: QueryRequest,
    api_key: str = Security(rate_limit_dependency)
):
    """
    Queries the agentic RAG system. Enforces rate limits and API Key security.
    Returns response answer, citations, confidence rating, and metrics.
    """
    # Reject extremely long query edge case
    if len(request.query.strip()) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is too long. Limit queries to 1000 characters."
        )

    try:
        # Execute query pipeline
        result = pipeline.query(request.query, filter_dict=request.metadata_filter)
        
        # Format metrics dictionary for structured JSON logging
        metrics = {
            "query": request.query,
            "latency_ms": result["latency_ms"],
            "chunks_retrieved": len(result["chunks"]),
            "tokens_used": result["tokens_used"],
            "retries": result["retries"],
            "cost_estimate": result["cost"],
            "confidence": result["confidence"],
            "status": result["status"]
        }
        
        # Write structured JSON log to stdout
        logger.info(f"Query execution complete: '{request.query[:50]}...'", extra={"metrics": metrics})
        
        # Record metrics in-memory
        metrics_tracker.record_query(metrics)

        return QueryResponse(
            answer=result["answer"],
            status=result["status"],
            confidence=result["confidence"],
            retries=result["retries"],
            cost=result["cost"],
            latency_ms=result["latency_ms"],
            chunks=[
                ChunkDetail(
                    id=c["id"],
                    text=c["text"],
                    metadata=c["metadata"],
                    score=c["score"]
                ) for c in result["chunks"]
            ]
        )
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline error: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse, tags=["Diagnostics"])
def get_system_metrics(api_key: str = Security(get_api_key)):
    """
    Surfaces aggregated query telemetry metrics and recent query runs.
    Protected by API key authentication.
    """
    success_rate = 0.0
    if metrics_tracker.total_queries > 0:
        success_rate = (metrics_tracker.successes / metrics_tracker.total_queries) * 100.0
        
    avg_latency = 0.0
    if metrics_tracker.total_queries > 0:
        avg_latency = metrics_tracker.total_latency_ms / metrics_tracker.total_queries

    return MetricsResponse(
        uptime_seconds=time.time() - start_time,
        total_queries=metrics_tracker.total_queries,
        success_rate_percent=success_rate,
        average_latency_ms=avg_latency,
        total_tokens_used=metrics_tracker.total_tokens,
        total_cost_usd=metrics_tracker.total_cost,
        query_history=metrics_tracker.history
    )

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def get_frontend():
    """
    Serves the Groundwork single-page frontend.
    """
    return HTMLResponse(content=get_frontend_html(), status_code=200)

@app.post("/ingest_file", tags=["Ingestion"])
def ingest_single_file(
    file: UploadFile = File(...),
    api_key: str = Security(rate_limit_dependency)
):
    """
    Ingests a single uploaded file (PDF, HTML, MD) directly.
    """
    t_start = time.time()
    try:
        # Save file to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Extract filename
        original_filename = file.filename
        
        # Load the single file using loader
        from app.ingestion.loader import load_file
        doc = load_file(tmp_path)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported or unparseable file format: {original_filename}"
            )
            
        # Restore original filename in metadata
        doc.metadata["filename"] = original_filename
        doc.metadata["file_path"] = f"uploads/{original_filename}"
        
        # Chunk
        chunks = chunk_document(doc, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        
        # Deduplicate
        unique_chunks = deduplicate_chunks(chunks)
        
        # Upsert
        chunks_upserted = store.upsert_chunks(unique_chunks)
        
        # Return count
        return {
            "status": "success",
            "filename": original_filename,
            "chunks_added": chunks_upserted,
            "chunks_skipped": len(unique_chunks) - chunks_upserted,
            "time_taken_seconds": time.time() - t_start
        }
    except Exception as e:
        logger.error(f"File ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}"
        )
