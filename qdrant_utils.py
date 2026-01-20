from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
from config import api_config
import os

load_dotenv()

# Initialize the Qdrant client with the URL, API key, and a connection timeout
qdrant = QdrantClient(
    url=api_config.qdrant_url,
    api_key=api_config.qdrant_api_key,
    timeout=60.0 # Time out set to prevents application from getting stuck waiting
)

def create_collection_if_not_exists(collection_name: str, vector_size: int = 384):
    """
    Checks if a collection exists in Qdrant and creates it if it doesn't.
    
    Args:
        collection_name (str): The name of the collection to check/create.
        vector_size (int): The dimension of the vectors to be stored in the collection. Defaults to 1024.
    """
    # Get the list of all collections currently in the Qdrant instance
    collections = qdrant.get_collections().collections

    # Check if a collection with the given name already exists in the list
    exists = any(c.name == collection_name for c in collections)

    # If the collection does not exist, create it
    if not exists:
        # Create a new collection with the specified name and vector configuration
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created")
    else:
        print(f"Collection '{collection_name}' already exists")

    return qdrant
