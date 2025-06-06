FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV APP_NAME=rag_chat_api \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    #    GOOGLE_API_KEY and GITHUB_TOKEN should be passed securely at runtime
    CHROMA_PATH=/app/db/chroma \
    DATA_STORE_PATH=/app/db/knowledge_base 

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/db/chroma && \
    chmod -R 755 /app/db

# Copy and install requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-built ChromaDB (if it exists locally)
COPY db/chroma /app/db/chroma
RUN chmod -R 755 /app/db/chroma

# Copy configuration files
COPY .env.prod /app/.env.prod
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Copy source code
COPY embeddings.py api.py /app/

# Expose the API port
EXPOSE 8000

# Command to run the application
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]