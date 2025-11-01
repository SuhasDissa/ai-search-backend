import os
from typing import List

class Settings:
    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2-0.5B-Instruct")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # API settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # RAG settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    DOCUMENTS_DIR: str = "/app/data/documents"

    # Web search settings
    WEB_SEARCH_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 5

    # Paths
    FAISS_INDEX_PATH: str = "/app/data/faiss_index"

settings = Settings()
