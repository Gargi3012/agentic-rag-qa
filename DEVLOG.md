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

## 📝 Candidate Reflection & Reflection Placeholders
*(Note: As required by the submission guidelines, the final candidate reflections must be written manually by the candidate).*

- **Reflections on Development**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
- **AI Tool Usage Disclosure**: `[TO BE FILLED BY CANDIDATE — do not use AI for this section]`
