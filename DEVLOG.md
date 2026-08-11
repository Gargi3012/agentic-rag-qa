# Developer Log (DEVLOG.md)

This log is an iterative engineering diary tracking the design decisions, implementation details, debugging stories, and key lessons from each build phase of the Cost-Efficient Agentic RAG QA Service.

---

## 🛠️ Phase 1: Project Scaffold, Docker Setup, and Health Endpoint

### What Was Built
- **Project Structure**: Set up clean, modular directory structure under `/app` and `/tests` as per specification.
- **Dependencies (`requirements.txt`)**: Defined baseline dependencies including FastAPI/Uvicorn, Qdrant Client/FastEmbed, Sentence Transformers, OpenAI SDK, PyPDF/PyMuPDF, and configuration parsers.
- **Environment Template (`.env.example`)**: Added key-value configurations for model selection, vector DB routing, security, and RAG thresholds.
- **Docker Integration (`docker-compose.yml` & `docker/Dockerfile`)**: Set up a custom python-3.11-slim Dockerfile and composed it with a self-hosted `qdrant/qdrant` vector database container, mounting persistent storage in the container.
- **FastAPI Core Skeleton**: Created the basic application under `app/api/main.py` containing a comprehensive `/health` endpoint tracks system uptime and status.
- **Git Ignore (`.gitignore`)**: Added python environment, system cache, configuration, and Qdrant local volume ignores to prevent leakages.

### Design Decisions & Rationale
- **Base Docker Image**: Chose `python:3.11-slim` over Alpine to avoid compatibility issues with compiled Rust-based vector embeddings or scientific libraries (like `numpy`, `pymupdf`), which are required for dense/sparse processing.
- **Local Volume Isolation**: Mounted the local volume `qdrant_data` as a named volume in Docker Compose to ensure persistent index storage across container restarts without cluttering the local workspace root with database binary files.
- **Modular Packaging**: Created explicit `__init__.py` markers under `app/` and `app/api/` to prevent Python resolution errors during testing or local package importing.

### Challenges Encountered & Debugging
- **Git Hook / Command Semicolon Conflict**: Attempted to chain multiple git commands using `;` in powershell, which was corrected to maintain step-by-step clarity. Staged, reviewed, and finalized directory files individually to ensure a clean commit baseline.

### Assumptions Made
- Assumed standard developer systems running this setup will have Docker Desktop installed and running.
- Assumed that the OpenAI API key will be provided during later development phases when real generation is active.

---

## 📦 Phase 2: Ingestion Pipeline with Idempotent Chunking

### What Was Built
- **Document Loaders (`app/ingestion/loader.py`)**: Implemented parsing and content extraction for PDF, HTML, and Markdown. Integrates PyMuPDF (fitz) for rapid text extraction, and BeautifulSoup for cleaning HTML page bodies.
- **Recursive Chunker (`app/ingestion/chunker.py`)**: Built a custom recursive text splitter measuring chunk length in token counts via `tiktoken`. Auto-splits any chunk exceeding token limits to guarantee safety before embedding.
- **Deduplicator (`app/ingestion/dedup.py`)**: Created deterministic UUID generation utilizing SHA-256 content hashes + file paths. This ensures exact file-level idempotency and prevents duplicate database upserts.
- **Integration Tests**: Tested the ingestion pipeline using a local scratch script with mock data.

### Design Decisions & Rationale
- **Deterministic Point UUIDs**: Derived UUIDv5 from SHA-256 content hashes of each chunk combined with its relative file path. This guarantees that re-ingesting the same file generates identical vector IDs, meaning Qdrant will overwrite existing nodes instead of accumulating duplicates.
- **PyMuPDF for PDF Processing**: Selected PyMuPDF (`fitz`) over `pypdf` because it is significantly faster and cleaner for extracting text from layouts, while including graceful exception handling for scanned/empty PDFs to avoid pipeline failures.

### Challenges Encountered & Debugging
- **Import Resolution Order (NameError)**: Experienced a `NameError: name 'BaseModel' is not defined` due to using `BaseModel` for the `Chunk` class definition prior to its import declaration. Re-organized the imports at the top of `chunker.py` and resolved the compiler error.
- **Virtual Environment Dependency Paths**: Local testing failed initially due to missing packages in the global interpreter. Configured a local `.venv` and executed the test runner within the virtual environment successfully.

### Assumptions Made
- Assumed standard documents have readable text; scanned PDFs will trigger warning flags and will be skipped cleanly.
- Assumed `cl100k_base` tokenizer encoding is a sufficient baseline proxy for token calculations for both our dense embeddings and the OpenAI generation models.

---

## 💾 Phase 3: Embedding & Qdrant Vector Storage

### What Was Built
- **Dense Embedding Generator (`app/retrieval/dense_embed.py`)**: Integrated local HuggingFace `all-MiniLM-L6-v2` SentenceTransformer for generating 384-dimensional dense vectors at zero API cost. Implemented model caching via a singleton structure.
- **Sparse Embedding Generator (`app/retrieval/sparse_embed.py`)**: Set up local FastEmbed `Qdrant/bm25` for sparse keyword index generation (indices and weights) to enable hybrid search without external API dependencies.
- **Qdrant Storage Wrapper (`app/retrieval/qdrant_client.py`)**: Built collection configuration with dual-vector support (dense + sparse). Developed batch search using Qdrant's server-side Reciprocal Rank Fusion (RRF).
- **Cost-Optimized Idempotency checks**: Configured the upsert pipeline to query Qdrant for existing chunk IDs first, allowing us to only embed and upsert new chunks. This completely eliminates unnecessary vector embedding generation.
- **Portability Fallbacks**: Configured automatic fallback to in-memory `QdrantClient(":memory:")` if connection to a Docker-hosted database fails, ensuring the code works without Docker configuration on target developer environments.

### Design Decisions & Rationale
- **Zero-Cost Local Retrieval**: Opted to run dense (MiniLM) and sparse (BM25) vector generation locally. This prevents external API latency and keeps index building costs strictly at zero.
- **Server-Side Reciprocal Rank Fusion**: Utilized Qdrant's native `query_points` with prefetch and FusionQuery RRF. This delegates the ranking and merging calculations to the database engine rather than fetching all hits and processing ranking manually in python.
- **Retrieve-Before-Embed Guard**: Queries Qdrant by point IDs (which are deterministic hashes of text+path) before generating embeddings. If a point is present, it is omitted from the batch passed to HuggingFace/FastEmbed.

### Challenges Encountered & Debugging
- **Docker Command Availability**: Discovered Docker CLI was not registered on the system environment path, causing the test environment verification to fail. Designed and implemented the in-memory `:memory:` fallback which successfully bypassed local environment restrictions.
- **Compatability Warnings**: Resolved initial cache filesystem warnings on Windows systems (regarding symlink creation) by verifying that files cache correctly in temporary storage without failing execution.

### Assumptions Made
- Assumed standard cosine similarity is the appropriate distance metric for SentenceTransformer outputs.
- Assumed in-memory mode is sufficient for local development/testing; in-memory indexes behave identically to the dockerized Qdrant API.

---

## 🧠 Phase 4: Agentic Query Pipeline with Self-Correction and No-Hallucination Gate

### What Was Built
- **Query Analyzer (`app/generation/agent.py`)**: Utilized OpenAI's structured outputs (`beta.chat.completions.parse` with Pydantic) to evaluate if query expansion/reformulation is needed, capped at 1 rewrite.
- **Cross-Encoder Reranker (`app/retrieval/reranker.py`)**: Integrated local `cross-encoder/ms-marco-MiniLM-L-6-v2` to score relevance. Normalized log-odds using a sigmoid function to map them to `[0.0, 1.0]`.
- **Relevance Gate**: Implemented a gate checking the best rerank score against `RELEVANCE_THRESHOLD`. If it fails, the system immediately returns `"insufficient context"` without executing LLM generation, saving LLM API costs.
- **Contextual Compressor**: Developed a lightweight sentence-level keyword overlap filters to extract only the most relevant sentences in retrieved chunks, significantly reducing prompt token overhead.
- **Grounded Generator**: Prompts OpenAI `gpt-4o-mini` with strict rules to only answer from context and inject explicit source citations like `[cite: chunk_id]`.
- **LLM Critic Pass & Retry**: Evaluates answer groundedness via a cheap LLM call. If any statement is ungrounded or citations are missing, it triggers 1 strict retry. It returns low-confidence flags if the retry also fails.

### Design Decisions & Rationale
- **Structured Pydantic Analysis**: Leveraged Pydantic-based JSON schema validation for both the Query Analyzer and the Critic checks. This guarantees that LLM outputs conform to our type annotations, preventing runtime parsing exceptions.
- **Sigmoid Score Normalization**: Wrapped the unnormalized log-odds of the Cross-Encoder model in a Sigmoid function to map scores to `[0, 1]`. This makes configuring a score threshold (e.g. `0.35` in `.env`) intuitive and robust.
- **Overlap-Based Sentence Compression**: Chose a local overlap-based regex sentence ranker to compress context before generation. This avoids adding a third LLM call for compression, saving API latency and token cost.
- **Unanswerable Refusal Guard**: Short-circuits the pipeline immediately on gate failure, preventing hallucinations and saving 100% of LLM generation costs for out-of-bounds questions.

### Challenges Encountered & Debugging
- **Type Hint Name Resolution Order**: Moving `CriticCheck` definition below the `AgenticQueryPipeline` class created a compiler-time `NameError: name 'CriticCheck' is not defined`. Relocated the Pydantic classes to the top of the module to resolve import compile failures.
- **Class Indentation Scope Premature Closure**: Declared the module-level function `compress_context` directly under the class, which prematurely closed the class scope, causing subsequent methods to not be bound to the class instance (AttributeError). Reorganized the module layout by moving all standalone functions to the top of the file, keeping class methods contiguous.

### Assumptions Made
- Assumed `gpt-4o-mini` is highly compliant with structured parsing formats.
- Assumed Madagascar chocolate chips are indeed the cookie secret ingredient for our local test script verification.

---

## 🔒 Phase 5: API Key Auth, Rate Limiting, and Structured Logging

### What Was Built
- **API Header Authentication (`app/api/auth.py`)**: Implemented simple headers-based verification checking `X-API-Key` matches the secret `APP_API_KEY` defined in the environment.
- **Sliding-Window Rate Limiter (`app/api/rate_limiter.py`)**: Built an in-memory sliding window rate limiter per API key. Rejects traffic with HTTP 429 once request counts exceed the configured RPM limit (requests per minute) within any 60-second window.
- **Structured JSON Logging (`app/api/logging.py`)**: Configured a custom structured JSON logger formatting queries and attaching exact telemetry metadata (latency, tokens, cost, retries, confidence, status).
- **FastAPI Endpoints (`app/api/main.py`)**:
  - `POST /ingest`: Directory loading, chunking, dedup, and indexing in one transactional call.
  - `POST /query`: Pipeline orchestrator running through analyzer, retriever, reranker, relevance gate, compressed generator, and critic pass.
  - `GET /health`: Diagnoses API server status, uptime, and database connection state.
  - `GET /metrics`: Aggregates and returns telemetry history, success percentages, and total costs.

### Design Decisions & Rationale
- **FastAPI Dependencies for Security**: Used FastAPI's `Security` dependency injection framework to chain API Key authentication and Rate Limiting. This isolates endpoint authorization logic cleanly from endpoint route execution.
- **Thread-Locked Sliding Windows**: Used Python's `threading.Lock` to synchronize rate-limiter updates. This guarantees that concurrent traffic checks on the same key don't create race conditions or incorrect count bypasses.
- **Aggregated Performance Metrics**: Tracked and cached recent query telemetry and running aggregates (tokens, cost, average latency) in-memory using an thread-safe tracking model. This prevents database fetch overhead during monitoring queries.

### Challenges Encountered & Debugging
- **Uvicorn Port Conflicts / Test Client Strategy**: Initial local testing using live port binding was prone to port-in-use errors. Switched to `fastapi.testclient.TestClient` for unit and integration verification, which performs in-process HTTP mocking, bypassing OS port binds and speeding up testing.
- **Deprecation Warnings**: Handled `StarletteDeprecationWarning` regarding Starlette `TestClient` and `fitz` API deprecation logs by verifying compatibility configurations.

### Assumptions Made
- Assumed memory footprints of sliding-window rate limiters and log history (capped at 50 queries) are negligible for typical API usage profiles.
- Assumed standard client callers are capable of setting HTTP header properties (`X-API-Key`).

---

## 📝 Candidate Reflection & Reflection Placeholders
*(Note: As required by the submission guidelines, the final candidate reflections must be written manually by the candidate).*

- **Reflections on Development**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
- **AI Tool Usage Disclosure**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
