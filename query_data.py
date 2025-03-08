import argparse
import logging
from typing import List, Tuple, Optional

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import Document

from get_embedding_function import get_embedding_function

# Configuration constants
CHROMA_PATH = "chroma"
DEFAULT_MODEL = "deepseek-r1"
DEFAULT_RESULTS_COUNT = 5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# RAG prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def setup_argument_parser() -> argparse.Namespace:
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query a RAG system with your question.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--results", type=int, default=DEFAULT_RESULTS_COUNT,
                        help=f"Number of results to retrieve (default: {DEFAULT_RESULTS_COUNT})")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return parser.parse_args()


def get_similar_documents(query_text: str, k: int = DEFAULT_RESULTS_COUNT) -> List[Tuple[Document, float]]:
    """
    Retrieve documents similar to the query from the vector database.
    
    Args:
        query_text: The text to search for
        k: Number of results to return
        
    Returns:
        List of (document, similarity_score) tuples
    """
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB
        results = db.similarity_search_with_score(query_text, k=k)
        logger.debug(f"Retrieved {len(results)} documents from vector DB")
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise


def generate_response(query_text: str, context_text: str, model_name: str) -> str:
    """
    Generate a response to the query based on the context using the specified LLM.
    
    Args:
        query_text: The user query
        context_text: The context retrieved from the vector DB
        model_name: Name of the Ollama model to use
        
    Returns:
        Generated response text
    """
    try:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        logger.debug(f"Using model: {model_name}")
        model = OllamaLLM(model=model_name)
        
        return model.invoke(prompt)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise


def query_rag(query_text: str, model_name: str = DEFAULT_MODEL, k: int = DEFAULT_RESULTS_COUNT) -> Optional[str]:
    """
    Perform a RAG query against the configured vector store.
    
    Args:
        query_text: The query text
        model_name: The LLM model to use
        k: Number of documents to retrieve
        
    Returns:
        Response text or None if an error occurred
    """
    try:
        # Retrieve similar documents
        results = get_similar_documents(query_text, k)
        
        if not results:
            logger.warning("No relevant documents found for the query")
            return "I couldn't find relevant information to answer your question."
        
        # Format context from retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Generate response
        response_text = generate_response(query_text, context_text, model_name)
        
        # Format output with sources
        sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        
        return response_text
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        print(f"An error occurred: {str(e)}")
        return None


def main():
    """Main entry point for the application."""
    args = setup_argument_parser()
    
    # Configure logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Execute RAG query
    query_rag(args.query_text, args.model, args.results)


if __name__ == "__main__":
    main()
