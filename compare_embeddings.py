import os
import logging
from typing import List, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_env_variable(var_name: str) -> str:
    """Retrieve an environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is not set.")
    return value


def initialize_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize the Google Generative AI Embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model=genai.Model.GEMINI_1_5,
        temperature=0.2,
        max_output_tokens=256,
        top_k=40,
        top_p=0.95,
    )


def load_vector_store(embeddings: GoogleGenerativeAIEmbeddings, db_path: str) -> Chroma:
    """Load the vector store."""
    return Chroma(
        embedding_function=embeddings,
        persist_directory=db_path,
        collection_name="knowledge_base",
    )


def perform_similarity_search(
    vector_store: Chroma, query_embedding: List[float], k: int = 5
) -> List[Dict[str, Any]]:
    """Perform a similarity search and return the results."""
    return vector_store.similarity_search_by_vector(
        query_embedding,
        k=k,
        filter=None,
        include=["distance", "metadata"],
    )


def main():
    try:
        load_dotenv()

        # Load environment variables
        chroma_db_path = get_env_variable("CHROMA_PATH")
        knowledge_base_dir = get_env_variable("DATA_STORE_PATH")

        # Initialize embeddings
        embeddings = initialize_embeddings()
        logging.info("Embeddings initialized successfully.")

        # Load evaluator
        evaluator = load_evaluator("langchain/eval/embeddings/embedding_similarity")
        if not evaluator:
            raise ValueError("Evaluator not found.")
        logging.info("Evaluator loaded successfully.")

        # Load vector store
        vector_store = load_vector_store(embeddings, chroma_db_path)
        logging.info("Vector store loaded successfully.")

        # Embed query
        query = "What do we mean by 'knowledge base'?"
        logging.info("Embedding query...")
        query_embedding = embeddings.embed_query(query)
        if not query_embedding:
            raise ValueError("Query embedding failed.")
        logging.info("Query embedded successfully.")

        # Perform similarity search
        logging.info("Performing similarity search...")
        results = perform_similarity_search(vector_store, query_embedding)
        if not results:
            raise ValueError("Similarity search failed.")
        logging.info("Similarity search completed successfully.")

        # Print results
        logging.info("Results:")
        for result in results:
            logging.info(
                f"Distance: {result['distance']}, Metadata: {result['metadata']}"
            )

        logging.info("Data store generated successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    logging.info("Main function executed successfully.")
