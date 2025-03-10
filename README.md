# Simple RAG System

A Retrieval-Augmented Generation (RAG) system for querying board game rules using vector embeddings and LLMs.

## Overview

This project demonstrates a simple implementation of a RAG system that:

1. Processes PDF documents containing board game rules
2. Splits them into chunks and generates vector embeddings
3. Stores these embeddings in a Chroma vector database
4. Allows users to query the system in natural language
5. Retrieves relevant context and generates accurate answers about game rules

## Requirements

- Python 3.8+
- Ollama (for embeddings and inference)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/simple-rag.git
cd simple-rag
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running on your system. Visit [Ollama's website](https://ollama.ai/) for installation instructions.

## Usage

### Populating the Database

Before querying, you need to populate the vector database with your documents:

```bash
python populate_database.py --data_path data --chunk_size 500 --chunk_overlap 50
```

Options:
- `--data_path`: Directory containing PDF documents (default: 'data')
- `--chunk_size`: Size of text chunks (default: 500)
- `--chunk_overlap`: Overlap between chunks (default: 50)
- `--clear_db`: Clear the existing database before adding new documents

### Querying the System

To ask questions about board game rules:

```bash
python query_data.py --query "How many players can play Monopoly?" --model mistral
```

Options:
- `--query`: The question you want to ask
- `--model`: The Ollama model to use (default: mistral)
- `--k`: Number of similar documents to retrieve (default: 4)

### Running Tests

To run the test suite:

```bash
pytest test_rag.py
```

## Project Structure

- `data/`: Contains PDF documents with board game rules
- `chroma/`: Vector database storage (created during database population)
- `populate_database.py`: Script to process documents and populate the database
- `query_data.py`: Script to query the RAG system
- `get_embedding_function.py`: Provides embedding functionality using Ollama
- `test_rag.py`: Tests for validating the RAG system's responses

## How it Works

1. **Document Processing**: PDFs are loaded, parsed, and split into chunks with overlaps
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in a Chroma vector database
4. **Retrieval**: When a query is received, the system finds semantically similar chunks
5. **Response Generation**: Retrieved context is sent to an LLM along with the query to generate a response

## Extending the System

To add more game rules:
1. Add PDFs to the `data/` directory
2. Run `populate_database.py` with the `--clear_db` flag if you want to rebuild the entire database

## License

MIT 