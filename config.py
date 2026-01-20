import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings."""
    gemini_key: str
    qdrant_url: str
    qdrant_api_key: str
    ollama_url: str
    ollama_base_url: str
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            gemini_key=os.getenv("GEMINI_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            ollama_url="http://localhost:11434/api/chat",
            ollama_base_url = "http://localhost:11434"
        )

@dataclass
class AppConfig:
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "llama3.1"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    collection_name: str = "recipe_book"

api_config = APIConfig.from_env()
app_config = AppConfig()