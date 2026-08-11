# Cost-Efficient Agentic RAG QA Service - Engineering Log

This engineering log tracks the design decisions, implementation details, debugging chronicles, and key lessons from each build milestone of the Cost-Efficient Agentic RAG QA Service.

---

## 🛠️ Milestone 1: Project Scaffold, Docker Setup, and Health Endpoint

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [Dockerfile](file:///d:/agentic_rag/docker/Dockerfile) | `NEW` | Optimized Python 3.11-slim base image, preconfigures OS libraries for PyMuPDF compilation. |
| [docker-compose.yml](file:///d:/agentic_rag/docker-compose.yml) | `NEW` | Orchestrates a self-hosted `qdrant/qdrant` vector database container, mounting persistent storage volume. |
| [requirements.txt](file:///d:/agentic_rag/requirements.txt) | `NEW` | Declares dependencies including FastAPI/Uvicorn, Qdrant Client, SentenceTransformers, Tiktoken, and Pydantic. |
| [app/api/main.py](file:///d:/agentic_rag/app/api/main.py) | `NEW` | Core FastAPI application skeleton, diagnostics `/health` router tracking database state & system uptime. |

### 📐 Design Decisions & Architectural Trade-offs
- **Python 3.11-Slim vs. Alpine**: Selected `python:3.11-slim` over Alpine. Alpine frequently fails or requires lengthy build-essential compiles for scientific libraries like `numpy` and `pymupdf` (fitz). Slim strikes the ideal balance between low image size and compatibility.
- **Isolated Storage Volumes**: Mounted Qdrant storage into a named Docker volume (`qdrant_data`) rather than local file system binds, avoiding workspace pollution with database binary files.

> [!NOTE]  
> The API skeleton uses FastAPI's startup event hooks to test container connection states immediately on launch, preventing silent database disconnection states.

### 🐛 Challenges & Debugging Chronicles
- **PowerShell Semicolon Chaining**: Chaining git commands using semicolons failed on Windows PowerShell hosts. Resolved by executing the baseline project commit commands sequentially.

### 📋 Assumptions & Constraints
- Assumed development environments run Docker Desktop locally for compose orchestration.
- Assumed standard `.env` configuration template will be duplicated and filled by the developer.

---

## 📦 Milestone 2: Ingestion Pipeline with Idempotent Chunking

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [loader.py](file:///d:/agentic_rag/app/ingestion/loader.py) | `NEW` | Directory crawler with text parsers for PDF (PyMuPDF), HTML (BeautifulSoup), and Markdown files. |
| [chunker.py](file:///d:/agentic_rag/app/ingestion/chunker.py) | `NEW` | Recursive token-aware text splitter using `tiktoken` (cl100k_base) to manage context limits. |
| [dedup.py](file:///d:/agentic_rag/app/ingestion/dedup.py) | `NEW` | Idempotent duplicate prevention, deriving deterministic UUIDv5 from text hashes and file paths. |

### 📐 Design Decisions & Architectural Trade-offs
- **Token-Aware Splitting**: Measured chunk sizes by token count rather than character count, mirroring LLM context limits directly and avoiding truncation edge-cases.
- **Deduplication via Deterministic UUIDs**: Derived version-5 UUIDs using `hashlib.sha256(file_path + text)`. This guarantees that if the same file is re-ingested, the generated IDs remain identical. Qdrant performs an overwrite (upsert) instead of creating duplicate records.

> [!TIP]  
> Standardizing on PyMuPDF (`fitz`) over `pypdf` yields a 10x throughput improvement during text parsing of large documents.

### 🐛 Challenges & Debugging Chronicles
- **Import Resolution Order (NameError)**: Experienced a `NameError: name 'BaseModel' is not defined` due to declaring the Pydantic `Chunk` model before importing Pydantic modules. Corrected by reorganizing the module header import list.

### 📋 Assumptions & Constraints
- Assumed standard files contain readable text; scanned image-only PDFs are flagged and safely skipped during parsing.

---

## 💾 Milestone 3: Embedding & Qdrant Vector Storage

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [dense_embed.py](file:///d:/agentic_rag/app/retrieval/dense_embed.py) | `NEW` | Generates 384-dimensional dense vectors locally using cached HuggingFace `all-MiniLM-L6-v2`. |
| [sparse_embed.py](file:///d:/agentic_rag/app/retrieval/sparse_embed.py) | `NEW` | Generates BM25 sparse indices and frequency weights locally using FastEmbed. |
| [qdrant_client.py](file:///d:/agentic_rag/app/retrieval/qdrant_client.py) | `NEW` | Wrapper managing dual-vector schema setups, server-side RRF search fusion, and in-memory fallbacks. |

### 📐 Design Decisions & Architectural Trade-offs
- **Zero Embedding Cost**: Implemented dense and sparse generators locally using CPU-bound models (`all-MiniLM-L6-v2` and `FastEmbed/bm25`), keeping embedding-generation API costs at **$0.00**.
- **Server-Side Reciprocal Rank Fusion (RRF)**: Leveraged Qdrant's prefetch API to fuse dense and sparse search results directly in the database engine, avoiding python-side memory overhead.
- **Retrieve-Before-Embed Guard**: Checks Qdrant for existing chunk IDs before embedding new chunks. If a chunk already exists, it skips embedding and upsert entirely, saving substantial compute.

> [!WARNING]  
> If Docker is down, the client gracefully falls back to `QdrantClient(":memory:")`, allowing local tests to run without local container engines.

### 🐛 Challenges & Debugging Chronicles
- **Docker Daemon Absence**: Bypassed Qdrant server connection failures during tests on non-docker systems by designing a silent in-memory fallback.

---

## 🧠 Milestone 4: Agentic Query Pipeline with Self-Correction and Relevance Gating

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [agent.py](file:///d:/agentic_rag/app/generation/agent.py) | `NEW` | Agent orchestration: structured query analyzer, context compressor, citation generator, and critic loop. |
| [reranker.py](file:///d:/agentic_rag/app/retrieval/reranker.py) | `NEW` | Local `ms-marco-MiniLM-L-6-v2` Cross-Encoder reranker with sigmoid normalization mapping. |

### 📐 Design Decisions & Architectural Trade-offs
- **Sigmoid Score Normalization**: Normalizes Cross-Encoder log-odds to `[0.0, 1.0]` using `1 / (1 + exp(-x))` to make configuring the relevance gate threshold intuitive.
- **Relevance Gate**: Out-of-domain queries fail the relevance threshold and are rejected immediately with `"insufficient context"`, bypassing LLM generation and saving 100% of LLM cost.
- **Lightweight Context Compressor**: Ranks sentences in chunks matching query keywords. Limits chunks to the top 3 sentences, reducing prompt tokens and saving generation cost.
- **Critic Pass & Retry**: Cheap LLM critic checks for hallucinations. If it fails, it triggers exactly 1 retry with feedback.

```
       [User Query]
            │
    [Query Analyzer]
            │
    [Hybrid Retrieval]
            │
    [Cross-Encoder Rerank]
            │
     [Relevance Gate] ──(Failed)──> [Refusal: insufficient context]
            │
         (Passed)
            │
   [Context Compression]
            │
   [Grounded Generator]
            │
      [Critic Pass] ────(Failed)──> [Regenerate with feedback (Max 1)]
            │
         (Passed)
            │
     [Final Output]
```

### 🐛 Challenges & Debugging Chronicles
- **Class Indentation Scope Closure**: Placing a helper function under the class definition with 0-space indentation prematurely closed the class scope, causing subsequent methods to throw `AttributeError`. Corrected by moving helper functions to the top of the module.

---

## 🔒 Milestone 5: API Key Auth, Rate Limiting, and Structured Logging

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [auth.py](file:///d:/agentic_rag/app/api/auth.py) | `NEW` | API Key dependency verification checking headers for `X-API-Key`. |
| [rate_limiter.py](file:///d:/agentic_rag/app/api/rate_limiter.py) | `NEW` | Thread-safe sliding window rate-limiter per API key returning HTTP 429 on limit breaches. |
| [logging.py](file:///d:/agentic_rag/app/api/logging.py) | `NEW` | JSON-formatted structured logging mapping latency, tokens, cost, and retries. |
| [main.py](file:///d:/agentic_rag/app/api/main.py) | `MODIFY` | Exposes `/health`, `/ingest`, `/query`, and `/metrics` endpoints with dependency chains. |

### 📐 Design Decisions & Rationale
- **FastAPI Security Dependencies**: Used FastAPI's security dependency chain to enforce auth and rate limits, decoupling security logic from route execution.
- **Thread-Locked Sliding Windows**: Synchronized the sliding-window state transitions with a lock (`threading.Lock`) to prevent race conditions during concurrent request bursts.

### 🐛 Challenges & Debugging Chronicles
- **Port Conflict in Testing**: Live port binding in server verification scripts caused conflicts. Replaced with `fastapi.testclient.TestClient` for mock integrations to execute mock HTTP calls in-process.

---

## 📊 Milestone 6: Evaluation Harness & Metrics Benchmark

### 📋 Overview & Component Map

| Component / File | Status | Description |
| :--- | :---: | :--- |
| [dataset.json](file:///d:/agentic_rag/app/eval/dataset.json) | `NEW` | Fixed benchmark dataset of 18 QA pairs (15 on-domain, 3 out-of-domain) with target chunk IDs. |
| [metrics.py](file:///d:/agentic_rag/app/eval/metrics.py) | `NEW` | Computes Recall@1/5, Hit Rate, MRR, nDCG@5, Context Precision, EM, F1, and LLM-as-a-judge scores. |
| [harness.py](file:///d:/agentic_rag/app/eval/harness.py) | `NEW` | Benchmarking orchestrator, manages isolated collection recreation, run loops, and cost table generation. |

### 📐 Design Decisions & Rationale
- **Deterministic ID Casing Normalization**: Standardized path casing (`os.path.abspath(file_path).replace("\\", "/").lower()`) before hashing. This prevents Windows path drive casing (`d:\` vs `D:\`) from producing different deterministic UUIDs, aligning expected IDs with Qdrant points.
- **Isolated Evaluation Collection**: Configured the evaluation runner to use a dedicated collection `agentic_rag_eval` and recreate it from scratch. This guarantees a clean, unpolluted data environment on every benchmark execution.

### 🐛 Challenges & Debugging Chronicles
- **Windows Drive Casing Bug**: Discovered that a lowercase `d:\` in local loading scripts and an uppercase `D:\` resolved by relative path libraries in the harness resulted in different UUIDs, causing on-domain queries to score 0.0 recall. Resolved by applying path normalization before hashing.
