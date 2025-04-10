FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV APP_NAME=rag_chat_api
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p db/chroma && \
    chmod -R 755 db

# IMPORTANT: Copy and install requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-built ChromaDB (if it exists locally)
COPY db/chroma /app/db/chroma

# Set proper permissions for the ChromaDB
RUN chmod -R 755 /app/db/chroma

# Copy configuration files
COPY .env.prod ./.env.prod
COPY docker-entrypoint.sh ./
RUN chmod +x ./docker-entrypoint.sh

# Copy source code
COPY embeddings.py api.py ./

# Expose the API port
EXPOSE 8000

# Command to run the application
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]