import os
from dotenv import load_dotenv

# Load env file
load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    
    # Security
    APP_API_KEY = os.getenv("APP_API_KEY", "rag123")
    
    # Rate Limiting
    RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
    
    # Models
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    
    # Groq & LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # RAG Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.001"))
