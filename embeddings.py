import shutil
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
DATA_STORE_PATH = os.getenv("DATA_STORE_PATH")
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

def _validate_environment_variables():
    """Validates that required environment variables are set."""
    if not CHROMA_DB_PATH:
        raise ValueError("CHROMA_PATH environment variable is not set.")
    if not DATA_STORE_PATH:
        raise ValueError("DATA_STORE_PATH environment variable is not set.")
    logging.info("Environment variables validated.")

def load_documents(directory: str) -> list[Document]:
    """Loads documents from the specified directory."""
    logging.info(f"Loading documents from: {directory}")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    loader = DirectoryLoader(directory)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from: {directory}")
    return documents

def split_text(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Splits documents into smaller chunks."""
    logging.info(f"Splitting {len(documents)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_chroma_db(chunks: list[Document], persist_directory: str, embedding_model: str) -> Chroma:
    """Creates and persists a ChromaDB from the document chunks."""
    logging.info(f"Creating ChromaDB at: {persist_directory}")
    if os.path.exists(persist_directory):
        logging.warning(f"Deleting existing ChromaDB at: {persist_directory}")
        shutil.rmtree(persist_directory)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=persist_directory,
        collection_name="knowledge_base"
    )

    logging.info(f"ChromaDB created and persisted to: {persist_directory}")

    # Debug logging
    collection_stats = db._collection.count()
    logging.debug(f"ChromaDB collection contains {collection_stats} documents")
    
    if collection_stats == 0:
        logging.warning("ChromaDB is empty after creation. Please check the document loading and splitting process.")
    else:
        logging.info(f"ChromaDB contains {collection_stats} documents after creation.")

    return db

def generate_data_store(data_store_path: str, chroma_db_path: str, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    """Orchestrates the process of loading, splitting, and saving data to ChromaDB."""
    logging.info("Starting data store generation.")
    try:
        documents = load_documents(data_store_path)
        chunks = split_text(documents, chunk_size, chunk_overlap)
        create_chroma_db(chunks, chroma_db_path, embedding_model_name)
        logging.info("Data store generation complete successfully.")
    except Exception as e:
        logging.error(f"An error occurred during data store generation: {e}", exc_info=True)

if __name__ == "__main__":
    _validate_environment_variables()
    generate_data_store(
        DATA_STORE_PATH,
        CHROMA_DB_PATH,
        EMBEDDING_MODEL_NAME,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    logging.info("Data store generation script finished.")