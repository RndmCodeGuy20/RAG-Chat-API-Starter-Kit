"""# api.py"""

import os
import logging
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from embeddings import load_documents, split_text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
LLM_MODEL_NAME = "gemini-1.5-pro"
DATA_STORE_PATH = os.getenv("DATA_STORE_PATH")
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

# Initialize FastAPI app
app = FastAPI(
    title="DevDocs Chat API",
    description="Query your documentation with natural language",
    version="1.0.0",
)


# Create request model
class QueryRequest(BaseModel):
    """Request model for querying documentation."""

    query: str = Field(..., description="The question to ask about your documentation")
    max_tokens: Optional[int] = Field(1024, description="Maximum tokens in response")
    relevance_threshold: Optional[float] = Field(
        0.4, description="Minimum relevance score threshold"
    )
    k: Optional[int] = Field(5, description="Number of documents to retrieve")


# Create response model
class QueryResponse(BaseModel):
    """Response model for querying documentation."""

    answer: str
    sources: list[str] = []
    relevance_scores: list[float] = []


# Template for prompts
# Replace the current PROMPT_TEMPLATE with this enhanced version

PROMPT_TEMPLATE = """
You are a specialized technical documentation assistant for software developers.

## CONTEXT INFORMATION
{context}

## QUESTION
{question}

## INSTRUCTIONS
1. Answer ONLY based on the provided context above.
2. If the context contains the complete answer, provide a detailed and thorough response.
3. If the context contains partial information, answer with what's available and clearly indicate what information is missing.
4. If the answer isn't in the context at all, respond with: "Based on the available documentation, I don't have information about this specific topic."
5. Include relevant code examples from the context when applicable.
6. Format your answer for clarity:
   - Use markdown formatting for headings and lists 
   - Format code in appropriate code blocks with language specification
   - Break complex concepts into smaller sections
7. Do not reference external knowledge or make assumptions beyond what's provided in the context.
8. If technical steps are involved, present them as numbered steps.
9. If there are warnings or important notes in the context, highlight them clearly.
10. If the user interacts with you by greetings or thanks, respond politely but keep the focus on the documentation.

## ANSWER:
"""

# Add CORS middleware - ADD THIS CODE BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)


def detect_conversation_type(query: str) -> tuple[str, bool]:
    """
    Detect if the query is a greeting, farewell, or regular question.

    Args:
        query: The user's query string

    Returns:
        tuple: (message_type, needs_rag)
        - message_type: 'greeting', 'farewell', or 'question'
        - needs_rag: Whether RAG search is needed
    """
    # Normalize query
    query_lower = query.lower().strip()

    # Common greetings
    greetings = [
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "what's up",
        "how are you",
        "nice to meet you",
        "hi there",
        "hello there",
    ]

    # Common farewells
    farewells = [
        "bye",
        "goodbye",
        "see you",
        "later",
        "take care",
        "farewell",
        "have a good day",
        "have a nice day",
        "until next time",
        "thanks",
        "thank you",
        "thanks a lot",
        "appreciate it",
        "cya",
    ]

    # Check if query is just a greeting
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting + " "):
            return "greeting", False

    # Check if query is just a farewell
    for farewell in farewells:
        if query_lower == farewell or query_lower.startswith(farewell + " "):
            return "farewell", False

    # Otherwise it's a question that needs RAG
    return "question", True


# Initialize embedding model and ChromaDB
@app.on_event("startup")
async def startup_db_client():
    """Initialize the embedding model and ChromaDB on startup."""
    app.embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    try:
        app.chroma_db = Chroma(
            embedding_function=app.embedding_function,
            persist_directory=CHROMA_DB_PATH,
            collection_name="knowledge_base",
        )
        collection_stats = app.chroma_db._collection.count()
        logger.info("ChromaDB initialized with %s documents", collection_stats)

        if collection_stats == 0:
            logger.warning(
                "ChromaDB is empty. Please ensure documents are loaded correctly."
            )
    except Exception as e:
        logger.error("Error initializing ChromaDB: %s", e)
        raise e


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "Welcome to DevDocs Chat API. Use /query endpoint to ask questions."
    }


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """Endpoint to query the documentation."""
    try:
        # Get the query from the request
        query = request.query
        logger.info("Received query: %s", query)

        # Check if the query is a greeting or farewell
        message_type, needs_rag = detect_conversation_type(query)

        # Handle greeting
        if message_type == "greeting":
            return QueryResponse(
                answer="👋 Hello! I'm your technical documentation assistant. How can I help you with your development questions today?",
                sources=[],
                relevance_scores=[],
            )

        # Handle farewell
        if message_type == "farewell":
            return QueryResponse(
                answer="Thanks for using the documentation assistant. If you have more questions later, feel free to ask!",
                sources=[],
                relevance_scores=[],
            )

        # Get the relevant documents with relevance scores
        results = app.chroma_db.similarity_search_with_relevance_scores(
            query=query, k=request.k
        )

        if not results:
            logger.warning("No relevant documents found in ChromaDB.")

            # Check if query looks like a question about the documentation
            doc_related_keywords = [
                "documentation",
                "docs",
                "manual",
                "guide",
                "tutorial",
                "api",
                "reference",
            ]

            if any(keyword in query.lower() for keyword in doc_related_keywords):
                return QueryResponse(
                    answer="I don't have enough information about that in the documentation. You can try rephrasing your question, or check if your question is related to the available documentation topics.",
                    sources=[],
                    relevance_scores=[],
                )

            # More general fallback
            return QueryResponse(
                answer="I'm a technical documentation assistant focused on helping with questions about the documented topics. I don't have information about that specific topic in my knowledge base. Please ask a question related to the documentation content.",
                sources=[],
                relevance_scores=[],
            )

        # Filter documents by relevance score
        relevant_documents = []
        relevant_scores = []
        sources = []

        for doc, score in results:
            if score > request.relevance_threshold:
                relevant_documents.append(doc)
                relevant_scores.append(score)
                # Extract source information
                if doc.metadata and "source" in doc.metadata:
                    sources.append(os.path.basename(doc.metadata["source"]))
                else:
                    sources.append("unknown")
            else:
                logger.warning(
                    "Document with score %s is below threshold and will not be included.",
                    score,
                )

        if not relevant_documents:
            logger.warning("No relevant documents found after filtering by score.")
            return QueryResponse(
                answer="I don't have enough information about that in the documentation.",
                sources=[],
                relevance_scores=[],
            )

        # Format the context for the prompt
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in relevant_documents]
        )

        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        # Send the prompt to the Google Generative AI API
        model = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME, max_output_tokens=request.max_tokens
        )
        response = model.invoke(prompt)

        return QueryResponse(
            answer=response.content, sources=sources, relevance_scores=relevant_scores
        )

    except Exception as e:
        logger.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reload")
async def reload_chroma():
    """Endpoint to reload ChromaDB."""
    try:
        # First check if we can write to the directory
        logging.info("Checking ChromaDB directory permissions: %s", CHROMA_DB_PATH)

        # return {"message": "Reloading ChromaDB..."}

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)

        # Attempt to create a test file to verify write permissions
        test_file = os.path.join(os.path.dirname(CHROMA_DB_PATH), "test_write.txt")
        try:
            with open(test_file, "w") as f:
                f.write("Testing write permissions")
            os.remove(test_file)
            logging.info("Write permissions confirmed for ChromaDB directory")
        except (PermissionError, IOError) as e:
            logging.error("No write permissions for ChromaDB directory: %s", e)
            return {
                "error": "Permission denied",
                "message": f"""Cannot write to {CHROMA_DB_PATH}. 
Please check permissions or use a different directory.""",
            }

        # Now proceed with the reload
        if os.path.exists(CHROMA_DB_PATH):
            try:
                logging.warning("Deleting existing ChromaDB at: %s", CHROMA_DB_PATH)
                shutil.rmtree(CHROMA_DB_PATH)
            except PermissionError as e:
                logging.error("Permission error deleting ChromaDB: %s", e)
                return {
                    "error": "Permission denied",
                    "message": f"Cannot delete existing database at {CHROMA_DB_PATH}."
                    "Try running: sudo chmod -R 755 {CHROMA_DB_PATH}",
                }

        # Create the directory with proper permissions
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)

        # Load and process documents
        documents = load_documents(DATA_STORE_PATH)
        chunks = split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        logger.info(
            "Loaded %s documents and split into %s chunks.", len(documents), len(chunks)
        )

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        app.chroma_db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name="knowledge_base",
        )

        collection_stats = app.chroma_db._collection.count()
        logger.info("ChromaDB reloaded with %s documents", collection_stats)

        return {
            "message": f"ChromaDB reloaded successfully with {collection_stats} documents"
        }

    except Exception as e:
        logger.error("Error reloading ChromaDB: %s", e)

        # Provide helpful error message for common issues
        error_msg = str(e).lower()
        if "readonly database" in error_msg:
            return {
                "error": "Read-only database",
                "message": "The database is read-only. Try running these commands:",
                "commands": [
                    f"sudo chown -R $USER {CHROMA_DB_PATH}",
                    f"chmod -R 755 {CHROMA_DB_PATH}",
                    f"rm -f {CHROMA_DB_PATH}/*.lock",
                ],
            }

        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
