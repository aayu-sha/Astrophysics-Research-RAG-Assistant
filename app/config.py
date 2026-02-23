from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Astrophysics RAG Assistant"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    FAISS_INDEX_PATH: Path = PROCESSED_DIR / "faiss_index"
    
    # FREE Ollama Settings (Local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    
    # FREE HuggingFace Settings (Local Embeddings)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_CHUNKS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)