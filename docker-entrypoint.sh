#!/bin/bash
set -e

# Print Python and environment information
python --version
echo "Starting API in environment: $APP_NAME"

# Check for required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY is not set"
fi

# Print Python and environment information
python --version
echo "Starting API in environment: $APP_NAME"

# Check for the existence of ChromaDB
if [ -d "/app/db/chroma" ] && [ "$(ls -A /app/db/chroma)" ]; then
  echo "Found existing ChromaDB"
else
  echo "WARNING: ChromaDB not found. The API may need to generate embeddings on first use."
fi

# Wait for any dependent services (if added later)
# sleep 5

# Execute the CMD
exec "$@"