"""# api.py"""

import os
import logging
from pathlib import Path
import shutil
from typing import Optional
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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

    query: str = Field(...,
                       description="The question to ask about your documentation")
    max_tokens: Optional[int] = Field(
        1024, description="Maximum tokens in response")
    relevance_threshold: Optional[float] = Field(
        0.4, description="Minimum relevance score threshold"
    )
    k: Optional[int] = Field(5, description="Number of documents to retrieve")


# Create response model
class QueryResponse(BaseModel):
    """Response model for querying documentation."""

    answer: str
    sources: list[dict] = []
    relevance_scores: list[float] = []


# Template for prompts
# Replace the current PROMPT_TEMPLATE with this enhanced version

PROMPT_TEMPLATE = """
You are a specialized historian specializing in Indian History.

## CONTEXT INFORMATION
{context}

## QUESTION
{question}

## INSTRUCTIONS
1. Answer ONLY based on the provided context above.
2. If the context contains the complete answer, provide a detailed and thorough response.

## ANSWER:
"""


forDocs = """
3. Never say phrases like "based on the documentation," "according to the context," or "the information provided."
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
11. Answer directly as if this information is your own knowledge, not as if you're referencing documentation.
12. If you don't have enough information to answer confidently, suggest the user check specific relevant documentation pages (use the URLs in the context).
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

    # Log Secrets
    logger.debug("""
    Google API key: %s,
    Github Token: %s,
    """, os.getenv("GOOGLE_API_KEY"), os.getenv("GITHUB_TOKEN"))

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


# Create a directory for uploaded files if it doesn't exist
UPLOADS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


class ProcessingStatus(BaseModel):
    """Model for the processing status response."""
    job_id: str
    status: str
    message: str


# Dictionary to store background job statuses
processing_jobs = {}


def process_document_background(file_path: str, job_id: str):
    """Background task to process uploaded document and create embeddings."""
    try:
        processing_jobs[job_id] = {
            "status": "processing", "message": "Processing document and creating embeddings..."}

        # Create a temporary directory for processing
        temp_dir = os.path.join(UPLOADS_DIR, job_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Load document and create chunks
        documents = load_documents(file_path)
        chunks = split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        # Create embeddings and add to existing ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

        # Create a separate collection for this upload or add to existing
        collection_name = f"uploaded_{job_id}"

        # Create a new ChromaDB instance for this document
        doc_db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=os.path.join(CHROMA_DB_PATH, job_id),
            collection_name=collection_name,
        )

        # Update the main ChromaDB to include this collection
        # This is optional - you might want separate collections per document
        # Or merge them into your main knowledge base

        # Update job status
        processing_jobs[job_id] = {
            "status": "completed",
            "message": f"Successfully processed document with {len(chunks)} chunks",
            "collection": collection_name
        }

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        processing_jobs[job_id] = {
            "status": "failed", "message": f"Error: {str(e)}"}


@app.post("/upload", response_model=ProcessingStatus)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(None)
):
    """
    Upload a document (PDF) to create vector embeddings.

    The document will be processed in the background and its embeddings
    will be stored in ChromaDB for future queries.
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Create uploads directory if it doesn't exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Save the uploaded file
    file_extension = Path(file.filename).suffix.lower()

    if file_extension != '.pdf':
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported at this time"
        )

    file_path = os.path.join(UPLOADS_DIR, f"{job_id}{file_extension}")

    # Save the uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Initialize job status
    processing_jobs[job_id] = {"status": "queued",
                               "message": "Job queued for processing"}

    # Start background processing
    background_tasks.add_task(process_document_background, file_path, job_id)

    return ProcessingStatus(
        job_id=job_id,
        status="queued",
        message="Document upload successful. Processing started in the background."
    )


@app.get("/upload/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Check the status of a document processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=404, detail=f"Job ID {job_id} not found"
        )

    job_info = processing_jobs[job_id]

    return ProcessingStatus(
        job_id=job_id,
        status=job_info["status"],
        message=job_info["message"]
    )


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "Welcome to DevDocs Chat API. Use /query endpoint to ask questions."
    }


@app.post("/query/{job_id}", response_model=QueryResponse)
async def query_specific_document(job_id: str, request: QueryRequest):
    """Query a specific uploaded document by its job ID."""
    # Check if the job exists and is completed
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Document with ID {job_id} not found")
    
    job_info = processing_jobs[job_id]
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Document processing is not complete. Current status: {job_info['status']}"
        )
    
    try:
        # Get the specific ChromaDB collection for this document
        specific_db = Chroma(
            embedding_function=app.embedding_function,
            persist_directory=os.path.join(CHROMA_DB_PATH, job_id),
            collection_name=job_info.get("collection", f"uploaded_{job_id}")
        )
        
        # Get the query from the request
        query = request.query
        logger.info(f"Received query for document {job_id}: {query}")
        
        # Check if greeting/farewell
        message_type, needs_rag = detect_conversation_type(query)
        if message_type in ["greeting", "farewell"]:
            return QueryResponse(
                answer=("👋 Hello! I can answer questions about your uploaded document." 
                       if message_type == "greeting" else 
                       "Thanks for using the document assistant!"),
                sources=[],
                relevance_scores=[]
            )
        
        # Get relevant documents
        results = specific_db.similarity_search_with_relevance_scores(
            query=query, k=request.k
        )
        
        if not results:
            return QueryResponse(
                answer="I couldn't find relevant information in your uploaded document.",
                sources=[],
                relevance_scores=[]
            )
        
        # Process results same as in your existing query endpoint
        relevant_documents = []
        relevant_scores = []
        sources = []
        
        for doc, score in results:
            if score > request.relevance_threshold:
                relevant_documents.append(doc)
                relevant_scores.append(score)
                if doc.metadata and "source" in doc.metadata:
                    sources.append({
                        "source": doc.metadata["source"],
                        "title": doc.metadata.get("title", "unknown"),
                    })
                else:
                    sources.append({"source": "unknown", "title": "unknown"})
        
        if not relevant_documents:
            return QueryResponse(
                answer="I couldn't find sufficiently relevant information in your uploaded document.",
                sources=[],
                relevance_scores=[],
            )
        
        # Format context and generate response
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in relevant_documents]
        )
        
        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        
        # Generate response
        model = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME, max_output_tokens=request.max_tokens
        )
        response = model.invoke(prompt)
        
        return QueryResponse(
            answer=response.content, 
            sources=sources, 
            relevance_scores=relevant_scores
        )
        
    except Exception as e:
        logger.error(f"Error querying document {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query", response_model=QueryResponse)
# async def query_docs(request: QueryRequest):
#     """Endpoint to query the documentation."""
#     try:
#         # Get the query from the request
#         query = request.query
#         logger.info("Received query: %s", query)

#         # Check if the query is a greeting or farewell
#         message_type, needs_rag = detect_conversation_type(query)

#         # Handle greeting
#         if message_type == "greeting":
#             return QueryResponse(
#                 answer="👋 Hello! I'm your technical documentation assistant. How can I help you with your development questions today?",
#                 sources=[],
#                 relevance_scores=[],
#             )

#         # Handle farewell
#         if message_type == "farewell":
#             return QueryResponse(
#                 answer="Thanks for using the documentation assistant. If you have more questions later, feel free to ask!",
#                 sources=[],
#                 relevance_scores=[],
#             )

#         # Get the relevant documents with relevance scores
#         results = app.chroma_db.similarity_search_with_relevance_scores(
#             query=query, k=request.k
#         )

#         if not results:
#             logger.warning("No relevant documents found in ChromaDB.")

#             # Check if query looks like a question about the documentation
#             doc_related_keywords = [
#                 "documentation",
#                 "docs",
#                 "manual",
#                 "guide",
#                 "tutorial",
#                 "api",
#                 "reference",
#             ]

#             if any(keyword in query.lower() for keyword in doc_related_keywords):
#                 return QueryResponse(
#                     answer="I don't have enough information about that in the documentation. You can try rephrasing your question, or check if your question is related to the available documentation topics.",
#                     sources=[],
#                     relevance_scores=[],
#                 )

#             # More general fallback
#             return QueryResponse(
#                 answer="I'm a technical documentation assistant focused on helping with questions about the documented topics. I don't have information about that specific topic in my knowledge base. Please ask a question related to the documentation content.",
#                 sources=[],
#                 relevance_scores=[],
#             )

#         # Filter documents by relevance score
#         relevant_documents = []
#         relevant_scores = []
#         sources = []

#         for doc, score in results:
#             if score > request.relevance_threshold:
#                 relevant_documents.append(doc)
#                 relevant_scores.append(score)
#                 # Extract source information
#                 if doc.metadata and "source" in doc.metadata:
#                     sources.append({
#                         "source": doc.metadata["source"],
#                         "title": doc.metadata.get("title", "unknown"),
#                     })
#                 else:
#                     sources.append("unknown")
#             else:
#                 logger.warning(
#                     "Document with score %s is below threshold and will not be included.",
#                     score,
#                 )

#         if not relevant_documents:
#             logger.warning(
#                 "No relevant documents found after filtering by score.")
#             return QueryResponse(
#                 answer="I don't have enough information about that in the documentation.",
#                 sources=[],
#                 relevance_scores=[],
#             )

#         # Format the context for the prompt
#         context_text = "\n\n---\n\n".join(
#             [doc.page_content for doc in relevant_documents]
#         )

#         # Create the prompt
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query)

#         # Send the prompt to the Google Generative AI API
#         model = ChatGoogleGenerativeAI(
#             model=LLM_MODEL_NAME, max_output_tokens=request.max_tokens
#         )
#         response = model.invoke(prompt)

#         return QueryResponse(
#             answer=response.content, sources=sources, relevance_scores=relevant_scores
#         )

#     except Exception as e:
#         logger.error("Error processing query: %s", e)
#         raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reload")
async def reload_chroma():
    """Endpoint to reload ChromaDB."""
    try:
        # First check if we can write to the directory
        logging.info(
            "Checking ChromaDB directory permissions: %s", CHROMA_DB_PATH)

        # return {"message": "Reloading ChromaDB..."}

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)

        # Attempt to create a test file to verify write permissions
        test_file = os.path.join(os.path.dirname(
            CHROMA_DB_PATH), "test_write.txt")
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
                logging.warning(
                    "Deleting existing ChromaDB at: %s", CHROMA_DB_PATH)
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
            "Loaded %s documents and split into %s chunks.", len(
                documents), len(chunks)
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
