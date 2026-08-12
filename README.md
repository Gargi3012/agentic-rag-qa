# Cost-Efficient Agentic RAG QA Service

This repository implements a production-grade, highly cost-optimized **Agentic Retrieval-Augmented Generation (RAG) QA Service** using FastAPI, Qdrant, and OpenAI `gpt-4o-mini`. 

- **Vector Store Chosen**: **Qdrant** (Self-hosted via Docker / Qdrant Cloud).
- **One-line Why**: Selected Qdrant because it natively supports both dense vectors and FastEmbed sparse BM25 vectors, permitting server-side Reciprocal Rank Fusion (RRF) in a single high-performance API query.

Retrieval embedding cost is kept strictly at **$0.00** by generating dense embeddings (`all-MiniLM-L6-v2`) and sparse indices (BM25) locally. Generation cost is optimized via relevance gating (short-circuiting unanswerable questions) and sentence-level context compression.

---

## 📐 System Architecture Diagram

```text
                  [User Input Query]
                          │
                  [Query Analyzer]  (gpt-4o-mini structured rewrite, max 1)
                          │
                  [Hybrid Search]   (Dense MiniLM + Sparse BM25 via Qdrant RRF)
                          │
                [Cross-Encoder Rerank] (ms-marco-MiniLM-L-6-v2)
                          │
                  [Relevance Gate]  (Sigmoid score threshold check: 0.35)
                         / \
                        /   \
                 (Passed)   (Failed) ──> [Refusal: insufficient context]
                      /
                     /
           [Context Compressor]     (Sentence-level keyword overlap compression)
                    │
           [Grounded Generator]     (gpt-4o-mini with source [cite: chunk_id])
                    │
              [Critic Pass]         (gpt-4o-mini structured grounding check)
                     / \
                    /   \
                 (Pass) (Fail) ───> [Regenerate with feedback (Max 1 retry)]
                  /
                 /
          [Final Answer]            (telemetry metrics logged in JSON)
```

---

## 🚀 Key Features

- **Local Hybrid Search**: Combines local semantic search (dense SentenceTransformers) and keyword search (Qdrant FastEmbed BM25) fused server-side using **Reciprocal Rank Fusion (RRF)**.
- **Agentic Self-Correction Loop**:
  - **Query Analyzer**: Automatically rewrites/expands search terms using structured schemas.
  - **Cross-Encoder Reranking**: Locally scores relevance using `ms-marco-MiniLM-L-6-v2` mapped to `[0.0, 1.0]` probabilities via a sigmoid function.
  - **Relevance Gate**: Refuses out-of-domain queries immediately, avoiding LLM generation API costs.
  - **Contextual Compressor**: Sentence-level keyword overlap extractor reduces token size.
  - **Critic Pass & Retry**: Cheap critique LLM pass validates groundedness and triggers 1 strict retry on hallucination detection.
- **Enterprise Security & Telemetry**:
  - **API Header Authentication**: Protects endpoints using header-bound `X-API-Key` checks.
  - **Sliding-Window Rate Limiter**: Thread-safe memory sliding tracker per API key.
  - **Structured JSON Logging**: Custom logger capturing structured event traces and latency metrics.
  - **Telemetry Endpoints**: Exposes `/metrics` returning uptime, error counts, costs, and latencies.
- **Evaluation Harness**: Automated benchmarking measuring Recall@k, Hit Rate, MRR, nDCG@k, Context Precision, EM, F1, and LLM-as-a-judge faithfulness/relevance scores.

---

## 📂 Project Structure

```text
├── app/
│   ├── api/
│   │   ├── auth.py           # Header API key validation
│   │   ├── logging.py        # Structured JSON logging formatter
│   │   ├── main.py           # FastAPI server routing & metrics telemetry
│   │   └── rate_limiter.py   # Thread-safe sliding window rate-limiter
│   ├── eval/
│   │   ├── dataset.json      # Ground truth benchmark dataset
│   │   ├── metrics.py        # Retrieval and generation metrics math
│   │   └── harness.py        # Evaluation harness orchestrator
│   ├── generation/
│   │   └── agent.py          # Query analyzer, generator, and critique loop
│   ├── ingestion/
│   │   ├── loader.py         # PDF (PyMuPDF), HTML (BS4), & Markdown parser
│   │   ├── chunker.py        # Token-based recursive character splitter
│   │   └── dedup.py          # Deterministic hash-based duplicate prevention
│   └── retrieval/
│       ├── dense_embed.py    # Local SentenceTransformer generator
│       ├── sparse_embed.py   # Local Qdrant FastEmbed BM25 generator
│       ├── reranker.py       # Local cross-encoder reranker
│       └── qdrant_client.py  # Dual-vector schema and RRF retrieval wrapper
├── data/                     # Sample document corpus files
├── docker/
│   └── Dockerfile            # Multi-stage python-slim deployment container
├── tests/
│   └── test_core.py          # Core unit test suite
├── docker-compose.yml        # Orchestrates Qdrant server container
├── requirements.txt          # Python project dependencies
├── DEVLOG.md                 # Iterative milestone log diary
├── eval_report.md            # Rendered evaluation benchmark report
└── eval_results.json         # Raw metrics results trace
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for self-hosted Qdrant server)

### 1. Setup Environment Configuration
Clone the repository and copy the environment template:
```bash
cp .env.example .env
```
Fill in the configuration details inside `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key (for query analysis, generation, and critique passes).
- `APP_API_KEY`: Secret string header token used to protect FastAPI endpoints (e.g. `rag123`).
- `QDRANT_URL`: Qdrant Cloud Cluster URL (optional).
- `QDRANT_HOST` / `QDRANT_PORT`: Local container coordinates (defaults to `localhost` and `6333`).
- `QDRANT_API_KEY`: Access Token key (needed only for Qdrant Cloud).
- `RATE_LIMIT_RPM`: Requests-per-minute threshold (e.g., `60`).

### 2. Run the FastAPI Application Locally
Start the server in reload mode (automatically defaults to in-memory Qdrant fallback if Docker is unavailable):
```bash
# Initialize and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI app
python -m uvicorn app.api.main:app --reload
```

Alternatively, to run the full stack including the self-hosted Qdrant database using Docker:
```bash
docker-compose up --build
```

---

## 🔌 API Usage Examples

All endpoints (except `/health`) require the `X-API-Key` header matching the `APP_API_KEY` defined in `.env`.

### 1. Document Ingestion
- **Route**: `POST /ingest`
- **Description**: Loads, chunks, deduplicates, and stores documents from a local directory. By default, documents are split into chunks of `512` tokens with a `50` token overlap (can be overridden inside `.env`).
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -H "X-API-Key: rag123" \
    -d "{\"directory_path\": \"d:\\\\agentic_rag\\\\data\"}"
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Ingestion completed.",
    "files_processed": ["qdrant_guide.md", "fastapi_tutorial.md", "rag_overview.md"],
    "total_chunks_indexed": 3
  }
  ```

### 2. Agentic Query
- **Route**: `POST /query`
- **Description**: Evaluates the query, retrieves and reranks context, gates out-of-context inputs, generates answers with explicit citations, and runs self-correction passes.
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -H "X-API-Key: rag123" \
    -d "{\"query\": \"How do you enable hybrid search in Qdrant?\"}"
  ```
- **Response**:
  ```json
  {
    "answer": "To enable hybrid search in Qdrant, you must configure both dense and sparse vector indexes. Dense search models semantic meaning using cosine distance, while sparse search models keyword match frequencies like BM25 [cite: 20da95a1-d819-57a5-bbe0-3bfc49e60ab9].",
    "chunks": [
      {
        "id": "20da95a1-d819-57a5-bbe0-3bfc49e60ab9",
        "text": "To enable hybrid search in Qdrant, you must configure both dense and sparse vector indexes. Dense search models semantic meaning using cosine distance, while sparse search models keyword match frequencies like BM25.",
        "metadata": {
          "filename": "qdrant_guide.md",
          "file_path": "d:\\agentic_rag\\data\\qdrant_guide.md"
        },
        "score": 0.9852
      }
    ],
    "confidence": "high",
    "retries": 0,
    "status": "success",
    "latency_ms": 4200.5,
    "cost": 0.000307,
    "tokens_used": 1668
  }
  ```

### 3. Telemetry Metrics
- **Route**: `GET /metrics`
- **Command**:
  ```bash
  curl -X GET http://localhost:8000/metrics \
    -H "X-API-Key: rag123"
  ```
- **Response**:
  ```json
  {
    "uptime_seconds": 1254.3,
    "total_queries": 15,
    "success_rate_percent": 93.33,
    "average_latency_ms": 3200.4,
    "total_tokens_used": 24500,
    "total_cost_usd": 0.00542,
    "query_history": [...]
  }
  ```

---

## 📊 Benchmarking & Testing

### Running Benchmark Evaluations
Calculates recall, MRR, nDCG, context precision, and LLM-as-a-judge scores over a 18-query dataset:
```bash
python app/eval/harness.py
```
Outputs are written to [eval_results.json](file:///d:/agentic_rag/eval_results.json) and [eval_report.md](file:///d:/agentic_rag/eval_report.md).

### Running Core Unit Tests
Runs the built-in python test suite:
```bash
python -m unittest tests/test_core.py
```

---

## 📐 Design Decisions & Architectural Trade-offs

### 1. Chunking Strategy & Rationale
We implemented a semantic-aware, token-budgeted recursive character splitter (`TokenRecursiveCharacterSplitter`). It splits text recursively by structural separators (first `\n\n`, then `\n`, then `" "`) to keep sentences and paragraphs intact. It counts splits in tokens using the `tiktoken` tokenizer (`cl100k_base` model) instead of characters, preventing context fragmentation while keeping chunks strictly within model token budgets.

### 2. Embedding Model Cost/Quality Trade-offs
To keep embedding costs at **$0.00**, we utilize local models:
- **Dense Vectors**: Cached HuggingFace `all-MiniLM-L6-v2` (384 dimensions) running locally on CPU.
- **Sparse Vectors**: FastEmbed `Qdrant/bm25` running locally on CPU.
*Trade-off*: Local models have slightly lower recall compared to paid commercial embeddings (e.g. OpenAI's `text-embedding-3-large`). We mitigate this by utilizing local Cross-Encoder reranking to ensure high relevance of the top-5 chunks before generation.

### 3. No-Hallucination Gate & Gating Loops
- **Relevance Gate**: The top-20 retrieved chunks are scored using `ms-marco-MiniLM-L-6-v2`. Raw scores are normalized to `[0.0, 1.0]` using a Sigmoid function. If the best score falls below `0.35`, the relevance gate fails and returns an unanswerable refusal immediately, saving 100% of LLM generation costs.
- **Grounded Generator**: Prompts require strict citations matching retrieved text chunk IDs.
- **Critic Pass**: A structured LLM check verifies that the answer is supported by the context. If it fails, it triggers 1 retry with critic feedback.

### 4. Idempotency Re-ingestion Guarantee
Chunk IDs are deterministic version-5 UUIDs generated from the SHA-256 hash of the chunk text combined with the file path. Windows paths are normalized (lowercase and forward-slashed) before hashing. Re-ingesting files overwrites existing points in Qdrant instead of accumulating duplicates, ensuring idempotency.

### 5. Self-Hosted vs. Managed DB Pricing Inflection Tiers
- **Small-Scale (<100K vectors)**: Managed serverless tiers (like Qdrant Cloud Free tier or Pinecone Serverless) are highly economical due to low minimum monthly fees.
- **Medium/Large Scale (>1M to 10M vectors)**: Running a self-hosted Qdrant instance inside Docker Compose on an AWS EC2 (e.g., `t3.medium` or `r6g.large`) yields **30% to 40% cost savings** compared to Qdrant Cloud starter/standard subscription plans.

### 6. Critic Self-Enhancement Bias
Because we use the same LLM (`gpt-4o-mini`) for both generating answers and critiquing them, there is an inherent risk of **self-enhancement bias** (the model tends to grade its own answers favorably). We mitigate this bias by enforcing strict JSON validation, structured Pydantic schema validation for the critic (`is_grounded` boolean), and providing clear natural-language reasoning requirements in system prompts.
