import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
LLM_MODEL_NAME = "gemini-1.5-pro"

# Initialize FastAPI app
app = FastAPI(
    title="DevDocs Chat API",
    description="Query your documentation with natural language",
    version="1.0.0", 
)

# Create request model
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about your documentation")
    max_tokens: Optional[int] = Field(1024, description="Maximum tokens in response")
    relevance_threshold: Optional[float] = Field(0.4, description="Minimum relevance score threshold")
    k: Optional[int] = Field(5, description="Number of documents to retrieve")

# Create response model
class QueryResponse(BaseModel):
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

# Initialize embedding model and ChromaDB
@app.on_event("startup")
async def startup_db_client():
    app.embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    try:
        app.chroma_db = Chroma(
            embedding_function=app.embedding_function,
            persist_directory=CHROMA_DB_PATH,
            collection_name="knowledge_base",
        )
        collection_stats = app.chroma_db._collection.count()
        logger.info(f"ChromaDB initialized with {collection_stats} documents")
        
        if collection_stats == 0:
            logger.warning("ChromaDB is empty. Please ensure documents are loaded correctly.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Welcome to DevDocs Chat API. Use /query endpoint to ask questions."}

@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    try:
        # Get the query from the request
        query = request.query
        logger.info(f"Received query: {query}")
        
        # Get the relevant documents with relevance scores
        results = app.chroma_db.similarity_search_with_relevance_scores(
            query=query, 
            k=request.k
        )
        
        if not results:
            logger.warning("No relevant documents found in ChromaDB.")
            return QueryResponse(
                answer="I don't have enough information about that in the documentation.",
                sources=[],
                relevance_scores=[]
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
                logger.warning(f"Document with score {score} is below threshold and will not be included.")
        
        if not relevant_documents:
            logger.warning("No relevant documents found after filtering by score.")
            return QueryResponse(
                answer="I don't have enough information about that in the documentation.",
                sources=[],
                relevance_scores=[]
            )
        
        # Format the context for the prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_documents])
        
        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        
        # Send the prompt to the Google Generative AI API
        model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, max_output_tokens=request.max_tokens)
        response = model.invoke(prompt)
        
        return QueryResponse(
            answer=response.content,
            sources=sources,
            relevance_scores=relevant_scores
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)