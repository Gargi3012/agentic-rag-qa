# Cost-Efficient Agentic RAG QA Service

This repository implements a production-grade, highly cost-optimized **Agentic Retrieval-Augmented Generation (RAG) QA Service** using FastAPI, Qdrant (Hybrid Search with Reciprocal Rank Fusion), and OpenAI `gpt-4o-mini`. 

Retrieval embedding cost is kept strictly at **$0.00** by generating dense embeddings (`all-MiniLM-L6-v2`) and sparse indices (BM25) locally. Generation cost is optimized via relevance gating (short-circuiting unanswerable questions) and sentence-level context compression.

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
- `RATE_LIMIT_RPM`: Requests-per-minute threshold (e.g., `60`).

### 2. Run the FastAPI Application Locally
Start the server in reload mode (automatically defaults to in-memory Qdrant fallback if Docker is unavailable):
```bash
# Initialize and activate virtual environment
python -m venv .venv
Source .venv/bin/activate  # On Windows: .venv\Scripts\activate

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

## 📊 Running Evaluations

The repository includes a self-contained evaluation harness that recreates a clean `agentic_rag_eval` collection, ingests sample documents from `data/`, executes 18 benchmark queries, and compiles analytics.

Execute the harness:
```bash
python app/eval/harness.py
```
This generates:
- **`eval_results.json`**: Raw execution metadata, token counts, costs, and pointwise evaluation metrics.
- **`eval_report.md`**: Rendered markdown summary reports displaying Recall, nDCG, latencies, and an infrastructure vector DB pricing comparison table.

---

## 🧪 Running Unit Tests

The test suite is built using Python's standard `unittest` library and is instantly runnable without external testing packages:
```bash
python -m unittest tests/test_core.py
```

---

## 🔌 API Documentation

All secured endpoints require the `X-API-Key` header matching the `APP_API_KEY` environment value.

### 1. Health Diagnostics
- **Route**: `GET /health` (Public)
- **Description**: Verifies API server uptime and Qdrant backend connectivity.
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": 1786467170.07,
    "version": "0.1.0",
    "uptime_seconds": 3600.0,
    "qdrant_connected": true
  }
  ```

### 2. Document Ingestion
- **Route**: `POST /ingest` (Secured)
- **Description**: Ingests all Markdown, HTML, and PDF documents within the specified directory path.
- **Body**:
  ```json
  {
    "directory_path": "d:\\agentic_rag\\data"
  }
  ```

### 3. Agentic Query
- **Route**: `POST /query` (Secured)
- **Description**: Orchestrates the query analysis, hybrid RRF search, reranking, relevance gating, context compression, grounded generation, and critic self-correction.
- **Body**:
  ```json
  {
    "query": "How do you enable hybrid search in Qdrant?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "To enable hybrid search in Qdrant, you must configure both dense and sparse vector indexes...",
    "chunks": [
      {
        "id": "20da95a1-d819-57a5-bbe0-3bfc49e60ab9",
        "text": "...",
        "metadata": { "filename": "qdrant_guide.md" },
        "score": 0.985
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

### 4. Observability Metrics
- **Route**: `GET /metrics` (Secured)
- **Description**: Returns running aggregates, average request latency, and a log of the last 50 queries.
