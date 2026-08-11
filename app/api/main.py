import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Cost-Efficient Agentic RAG QA Service",
    description="An agent-driven RAG QA API leveraging self-hosted Qdrant, dense+sparse embeddings, and gpt-4o-mini with self-correction loops.",
    version="0.1.0",
)

# Enable CORS for frontend/development access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime_seconds: float

start_time = time.time()

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Diagnostics endpoint to verify API server status and uptime.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="0.1.0",
        uptime_seconds=time.time() - start_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
