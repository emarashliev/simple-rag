import os
import logging
from typing import Optional
from functools import lru_cache

from langchain_ollama.embeddings import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)

# Default model to use for embeddings
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

# Get model name from environment variable if set
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_embedding_function(model_name: Optional[str] = None) -> OllamaEmbeddings:
    """
    Returns a configured embedding function using Ollama.
    
    This function is cached to avoid recreating the embedding model unnecessarily.
    
    Args:
        model_name: Name of the embedding model to use. 
                   If None, uses the EMBEDDING_MODEL environment variable or default.
    
    Returns:
        An initialized OllamaEmbeddings instance
    
    Raises:
        ConnectionError: If unable to connect to the Ollama service
        Exception: For other unexpected errors
    """
    try:
        # Use provided model_name, or fall back to environment variable/default
        embedding_model = model_name or EMBEDDING_MODEL
        
        logger.debug(f"Initializing embedding model: {embedding_model}")
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Test the embeddings with a simple string to verify it works
        _ = embeddings.embed_query("test")
        logger.debug(f"Successfully initialized embedding model: {embedding_model}")
        
        return embeddings
    
    except ConnectionError as e:
        logger.error(f"Failed to connect to Ollama service: {str(e)}")
        raise ConnectionError(f"Could not connect to Ollama. Is it running? Error: {str(e)}") from e
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise
