from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
from config import api_config
import os
import chromadb
from pathlib import Path

load_dotenv()

class QdrantDB():
    def __init__(self):
        # Initialize the Qdrant client with the URL, API key, and a connection timeout
        self.type = "qdrant"

        self.client = QdrantClient(
            url=api_config.qdrant_url,
            api_key=api_config.qdrant_api_key,
            timeout=60.0 # Time out set to prevents application from getting stuck waiting
        )

        self.client.get_collections()

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 384):
        """
        Checks if a collection exists in Qdrant and creates it if it doesn't.
        
        Args:
            collection_name (str): The name of the collection to check/create.
            vector_size (int): The dimension of the vectors to be stored in the collection. Defaults to 1024.
        """
        # Get the list of all collections currently in the Qdrant instance
        collections = self.client.get_collections().collections

        # Check if a collection with the given name already exists in the list
        exists = any(c.name == collection_name for c in collections)

        # If the collection does not exist, create it
        if not exists:
            # Create a new collection with the specified name and vector configuration
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            print(f"Collection '{collection_name}' created")
        else:
            print(f"Collection '{collection_name}' already exists")

    def del_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name=collection_name)
            print("Succesfully delete collection")
        except Exception as e:
            print("⚠️  Failed to delete collection")

    def document_exists(self, collection_name: str, doc_source: str) -> bool:
        """Check if vectors with specific source metadata exist."""
        try:
            scroll_result, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=doc_source)
                        )
                    ]
                ),
                limit=1
            )
            return len(scroll_result) > 0
        except Exception as e:
            print(f"⚠️ Check failed: {e}")
            return False

class LocalChromaDB:
    """Handler for Local ChromaDB (Fallback)"""
    def __init__(self):
        self.type = "chroma"
        self.persist_dir = Path(__file__).resolve().parent / "chroma_db_data"
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        print("📂 Initialized Local ChromaDB")

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 384):
        self.client.get_or_create_collection(name=collection_name)
        print(f"✅ Local Chroma collection '{collection_name}' ready")

    def del_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name)
            print("🗑️  Local Chroma collection deleted")
        except Exception as e:
            print(f"Collection clean: {e}")

    def document_exists(self, collection_name: str, doc_source: str) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
            existing = collection.get(where={"source": doc_source}, limit=1)
            return len(existing['ids']) > 0
        except Exception:
            return False
    

def get_db_instance():
    try:
        db = QdrantDB()
        print("Connected to Qdrant Cloud")
        return db
    except Exception as e:
        print(f"⚠️ Qdrant Connection Failed: {e}")
        print("🔄 Switching to Local ChromaDB...")
        return LocalChromaDB()
