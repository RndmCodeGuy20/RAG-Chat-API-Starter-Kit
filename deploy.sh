#!/bin/bash
set -e

# Prepare the database for inclusion in Docker image
./prepare-db.sh

# Build and start the Docker containers
# docker-compose up -d

docker buildx build \
  --platform linux/amd64 \
  -t rndmcodeguy/rag-chat-api:v1.0.2 \
  --build-arg GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --build-arg GITHUB_TOKEN=$GITHUB_TOKEN \
  --build-arg CHROMA_PATH=$CHROMA_PATH \
  --build-arg DATA_STORE_PATH=$DATA_STORE_PATH \
  --push . \

echo "API is running at http://localhost:8000"
echo "To test it, try: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"How to install rcg?\"}'"