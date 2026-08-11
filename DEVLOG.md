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

## 📝 Candidate Reflection & Reflection Placeholders
*(Note: As required by the submission guidelines, the final candidate reflections must be written manually by the candidate).*

- **Reflections on Development**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
- **AI Tool Usage Disclosure**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
