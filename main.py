import os
import logging

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
DATA_STORE_PATH = os.getenv("DATA_STORE_PATH")
RESPONSE_FILE_PATH = os.path.join('', "response.md")

if not CHROMA_DB_PATH:
    raise ValueError("CHROMA_PATH environment variable is not set.")
if not DATA_STORE_PATH:
    raise ValueError("DATA_STORE_PATH environment variable is not set.")

# Set default LLM model
LLM_MODEL_NAME = "gemini-1.5-pro"  # Use gemini-pro as default model

PROMPT_TEMPLATE = """
    Answer the question as detailed as possible and with appropriate code examples from the provided context, 
    make sure to provide all the details, if the answer is not in the provided context just say, "Sorry, can't help you with that". 
    Strictly follow the context and do not add any extra information. Do not repeat the question in your answer.
    
    Context: {context} \n
    Question: {question} \n

    Answer:
    """

def ensure_directories_exist():
    """Ensure that required directories exist."""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(DATA_STORE_PATH, exist_ok=True)

def main():
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if the data directory has any files
    has_files = False
    if os.path.exists(DATA_STORE_PATH):
        for root, dirs, files in os.walk(DATA_STORE_PATH):
            if any(file.endswith(('.txt', '.md', '.pdf')) for file in files):
                has_files = True
                break
    
    if not has_files:
        logging.warning(f"No documents found in {DATA_STORE_PATH}. Please add some documents first.")
        return

    # Check if embeddings need to be generated
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        logging.info("No ChromaDB found. Running embedding generation...")
        # Import here to avoid circular imports
        from embeddings import generate_data_store, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
        generate_data_store(
            DATA_STORE_PATH,
            CHROMA_DB_PATH,
            EMBEDDING_MODEL_NAME,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    try:
        chroma_db = Chroma(
            embedding_function=embedding_function,
            persist_directory=CHROMA_DB_PATH,
            collection_name="knowledge_base",
        )

        # Debug: Check if ChromaDB has documents
        collection_stats = chroma_db._collection.count()
        logging.info(f"ChromaDB collection contains {collection_stats} documents")
        
        if collection_stats == 0:
            logging.warning("ChromaDB is empty. Please ensure documents are loaded correctly.")
            return

        # Example query
        query = "What do we mean by 'knowledge base'?"

        # Get the relevant documents with relevance scores
        results = chroma_db.similarity_search_with_relevance_scores(query=query, k=5)

        # Log the raw response
        logging.info(f"Raw ChromaDB Response: {len(results)} results found. With scores: {[result[1] for result in results]}")

        if not results:
            logging.warning("No relevant documents found in ChromaDB.")
            return

        # Extract the relevant documents and scores but only allow documents with score > 0.4
        relevant_documents = []
        relevant_scores = []

        for doc, score in results:
            if score > 0.4:
                relevant_documents.append(doc)
                relevant_scores.append(score)
            else:
                logging.warning(f"Document with score {score} is below threshold and will not be included.")
                continue
        
        if not relevant_documents:
            logging.warning("No relevant documents found after filtering by score.")
            return

        # Log the extracted documents and scores
        for i, doc in enumerate(relevant_documents):
            logging.info(f"Retrieved Document {i+1}:\nContent: {doc.page_content}\nScore: {relevant_scores[i]}")

        # Format the context for the prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_documents])

        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        logging.info(f"Formatted Prompt:\n{prompt}")

        # Send the prompt to the Google Generative AI API with proper model parameter
        try:
            # Initialize the model with required parameters
            model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
            response = model.invoke(prompt)
            logging.info(f"Response from Google Generative AI: {response}")

            # Log the response to response file
            response_file_path = os.path.join("responses", f"response-{response.id}.md")
            with open(response_file_path, "w") as response_file:
                response_file.write(response.content)
            logging.info(f"Response logged to {response_file_path}")
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error calling Google Generative AI: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Error in main: {e}, traceback: {e.__traceback__}")

if __name__ == "__main__":
    main()