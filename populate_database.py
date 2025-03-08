import argparse
import logging
import os
import shutil
from typing import List, Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function

# Configure constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.Namespace:
    """
    Set up and parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Populate vector database with document chunks")
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, 
                        help=f"Path to directory containing PDF documents (default: {DATA_PATH})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Size of text chunks in characters (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap between chunks in characters (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--db-path", type=str, default=CHROMA_PATH,
                        help=f"Path to Chroma database directory (default: {CHROMA_PATH})")
    
    return parser.parse_args()


def load_documents(data_path: str) -> List[Document]:
    """
    Load PDF documents from the specified directory.
    
    Args:
        data_path: Path to directory containing PDF files
        
    Returns:
        List of loaded documents
        
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no documents were found
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")
            
        logger.info(f"Loading documents from {data_path}")
        document_loader = PyPDFDirectoryLoader(data_path)
        documents = document_loader.load()
        
        if not documents:
            raise ValueError(f"No documents found in {data_path}")
            
        logger.info(f"Loaded {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise


def split_documents(
    documents: List[Document], 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into chunks for embedding and retrieval.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of document chunks
    """
    try:
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assign unique IDs to document chunks based on source, page, and position.
    
    This creates IDs like "data/document.pdf:6:2" where:
    - data/document.pdf is the source
    - 6 is the page number
    - 2 is the chunk index within that page
    
    Args:
        chunks: List of document chunks to process
        
    Returns:
        List of document chunks with IDs added to metadata
    """
    try:
        logger.info(f"Calculating chunk IDs for {len(chunks)} chunks")
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the chunk metadata
            chunk.metadata["id"] = chunk_id

        return chunks
        
    except Exception as e:
        logger.error(f"Error calculating chunk IDs: {str(e)}")
        raise


def add_to_chroma(chunks: List[Document], db_path: str = CHROMA_PATH) -> None:
    """
    Add document chunks to the Chroma vector database.
    Only adds chunks that don't already exist in the database.
    
    Args:
        chunks: List of document chunks to add
        db_path: Path to Chroma database directory
    """
    try:
        if not chunks:
            logger.warning("No chunks to add to database")
            return
            
        # Load the existing database
        logger.info(f"Connecting to Chroma database at {db_path}")
        db = Chroma(
            persist_directory=db_path, 
            embedding_function=get_embedding_function()
        )

        # Add IDs to chunks
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Get existing document IDs
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"Found {len(existing_ids)} existing documents in database")

        # Filter out chunks that already exist in the database
        new_chunks = [
            chunk for chunk in chunks_with_ids 
            if chunk.metadata["id"] not in existing_ids
        ]

        # Add new chunks to the database
        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new document chunks to database")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            logger.info("Successfully added chunks to database")
        else:
            logger.info("No new document chunks to add")
            
    except Exception as e:
        logger.error(f"Error adding chunks to database: {str(e)}")
        raise


def clear_database(db_path: str = CHROMA_PATH) -> None:
    """
    Clear the Chroma database by removing its directory.
    
    Args:
        db_path: Path to Chroma database directory
    """
    try:
        if os.path.exists(db_path):
            logger.info(f"Clearing database at {db_path}")
            shutil.rmtree(db_path)
            logger.info("Database cleared successfully")
        else:
            logger.info(f"No database found at {db_path} - nothing to clear")
            
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise


def main():
    """
    Main entry point for the database population script.
    """
    try:
        # Parse command line arguments
        args = setup_argument_parser()
        
        # Override constants with command line arguments
        data_path = args.data_path
        db_path = args.db_path
        chunk_size = args.chunk_size
        chunk_overlap = args.chunk_overlap
        
        # Reset the database if requested
        if args.reset:
            clear_database(db_path)

        # Load and process documents
        documents = load_documents(data_path)
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        add_to_chroma(chunks, db_path)
        
        logger.info("Database population completed successfully")
        
    except Exception as e:
        logger.error(f"Database population failed: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
