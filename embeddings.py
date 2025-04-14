import sys
import shutil
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
DATA_STORE_PATH = os.getenv("DATA_STORE_PATH")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
BASE_URL = "https://docs.nudgenow.com/"


def _validate_environment_variables():
    """Validates that required environment variables are set."""
    if not CHROMA_DB_PATH:
        raise ValueError("CHROMA_PATH environment variable is not set.")
    if not DATA_STORE_PATH:
        raise ValueError("DATA_STORE_PATH environment variable is not set.")
    logging.info("Environment variables validated.")


def load_documents(repository: str) -> list[Document]:
    """Loads documents from the specified repository."""
    logging.info(f"Loading documents from repository: {repository}")
    # if not os.path.exists(repository):
    #     raise FileNotFoundError(f"repository not found: {repository}")
    # loader = DirectoryLoader(repository, glob="**/*.md")
    loader = GithubFileLoader(
        repo="nudgenow/nudge-devdocs",
        branch="prod_main",
        file_filter=lambda file_path: file_path.endswith(
            ".md") and file_path.startswith("docs/"),
        directory=["docs/"],
        access_token=GITHUB_TOKEN,
    )
    documents = loader.load()

    import re
    # Process the metadata to remove numbering prefixes
    for doc in documents:
        original_path = doc.metadata['source'].split("docs/")[-1]
        filename = os.path.basename(doc.metadata["source"])

        # Split the path into components
        path_parts = original_path.split('/')

        # Clean each component by removing the numbering prefix
        cleaned_parts = [
            re.sub(r'^\d+[\-\.]', '', part)
            for part in path_parts
        ]

        # Reassemble the path
        cleaned_path = '/'.join(cleaned_parts)

        # Replace spaces with underscores
        cleaned_path = cleaned_path.replace(' ', '%20')

        # Create web URL (remove .md extension)
        web_path = os.path.splitext(cleaned_path)[0]
        web_url = f"{BASE_URL}{web_path}"

        # print(f"Cleaned path: {cleaned_path}")
        # print(f"Web URL: {web_url}\n")

        if "title" not in doc.metadata:
            clean_filename = re.sub(r'^\d+[\-\.]', '', filename)
            clean_title = os.path.splitext(clean_filename)[
                0].replace('-', ' ').title()
            doc.metadata['title'] = clean_title

        # Update the document metadata with the new web URL
        doc.metadata['source'] = web_url
        doc.metadata['path'] = web_url

    logging.info(
        f"Loaded {len(documents)} documents from repository: {repository}\n\n"
    )

    return documents


def split_text(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Splits documents into smaller chunks."""
    logging.info(
        f"Splitting {len(documents)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks


def create_chroma_db(
    chunks: list[Document], persist_directory: str, embedding_model: str
) -> Chroma:
    """Creates and persists a ChromaDB from the document chunks."""
    logging.info(f"Creating ChromaDB at: {persist_directory}")
    if os.path.exists(persist_directory):
        logging.warning(f"Deleting existing ChromaDB at: {persist_directory}")
        shutil.rmtree(persist_directory)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
        collection_name="knowledge_base",
    )

    logging.info(f"ChromaDB created and persisted to: {persist_directory}")

    # Debug logging
    collection_stats = db._collection.count()
    logging.debug(f"ChromaDB collection contains {collection_stats} documents")

    if collection_stats == 0:
        logging.warning(
            "ChromaDB is empty after creation. Please check the document loading and splitting process."
        )
    else:
        logging.info(
            f"ChromaDB contains {collection_stats} documents after creation.")

    return db


def generate_data_store(
    data_store_path: str,
    chroma_db_path: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Orchestrates the process of loading, splitting, and saving data to ChromaDB."""
    logging.info("Starting data store generation.")
    try:
        documents = load_documents(data_store_path)
        chunks = split_text(documents, chunk_size, chunk_overlap)
        create_chroma_db(chunks, chroma_db_path, embedding_model_name)
        logging.info("Data store generation complete successfully.")
    except Exception as e:
        logging.error(
            f"An error occurred during data store generation: {e}", exc_info=True
        )


if __name__ == "__main__":
    _validate_environment_variables()
    generate_data_store(
        DATA_STORE_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
    )
    logging.info("Data store generation script finished.")
