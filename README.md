# RAG Chat API Starter Kit

A RAG (Retrieval Augmented Generation) system that uses Google's Gemini models to provide contextual responses based on your documentation.

## Overview

RAG Chat API enables you to create a chatbot that answers questions based on your technical documentation. It uses:

- **Google Gemini models** for advanced embeddings and text generation
- **ChromaDB** for vector storage and similarity search
- **LangChain** for document loading, chunking, and prompt management

This system creates embeddings of your documentation, stores them in a vector database, and then retrieves relevant context when answering user queries.

## Features

- ✅ Load and process documents (.txt, .md, .pdf (expected to be introduced in v2.0.0))
- ✅ Split documents into manageable chunks with overlap
- ✅ Generate and store embeddings with Google's text-embedding-004 model
- ✅ Semantic search for relevant document chunks
- ✅ Contextual answers using Gemini 1.5 Pro
- ✅ Customizable chunk size and overlap
- ✅ Organized response storage

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd devdocs-chat-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .chat-env
   source .chat-env/bin/activate  # On Windows: .chat-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file with your configuration:
   ```
   CHROMA_PATH="db/chroma"
   GOOGLE_API_KEY="your-google-api-key-here"
   DATA_STORE_PATH="db/knowledge_base"
   ```

## Usage

### 1. Add your documentation

Place your documentation files in the directory specified by `DATA_STORE_PATH` (default: knowledge_base):

```bash
mkdir -p db/knowledge_base
cp your-docs/*.md db/knowledge_base/
```

### 2. Generate embeddings

Run the embeddings script to process your documentation and create vector embeddings:

```bash
python embeddings.py
```

This will:
- Load all documents from your knowledge base directory
- Split them into chunks of text
- Generate embeddings using Google's text-embedding-004 model
- Store the embeddings in ChromaDB

### 3. Query your documentation

Run the main script to interact with your documentation:

```bash
python main.py
```

By default, this will:
- Ensure your knowledge base and ChromaDB directories exist
- Check if there are documents to process
- Generate embeddings if needed
- Run a sample query ("Why do we need to use embeddings?")
- Find relevant document chunks based on the query
- Generate a contextual response using Gemini 1.5 Pro
- Save the response to the responses directory

## Customization

You can customize the behavior by modifying:

- **Query**: Change the example query in main.py
- **Chunking parameters**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in embeddings.py
- **Model**: Change the embedding or LLM model in the respective files
- **Prompt template**: Modify the `PROMPT_TEMPLATE` in main.py

## Project Structure

- main.py: Main script for querying the documentation
- embeddings.py: Script for generating and storing embeddings
- .env: Environment variables
- db: Directory for databases
  - `chroma/`: ChromaDB vector database
  - `knowledge_base/`: Your documentation files
- responses: Generated responses

## Requirements

- Python 3.9+
- Google API key with access to Gemini models
- Dependencies listed in requirements.txt

## Troubleshooting

- **Empty ChromaDB**: Make sure you have documents in your knowledge base directory and run embeddings.py
- **No responses**: Check that your Google API key is valid and has access to the required models
- **Missing directories**: The system will create required directories automatically

## License

[Your License Here]

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
Make sure to follow the code style and include tests for new features.