#!/bin/bash
set -e

# Fetch the latest docs from source


# Ensure the ChromaDB exists locally
if [ ! -d "db/chroma" ] || [ ! "$(ls -A db/chroma)" ]; then
  echo "No ChromaDB found. Generating embeddings..."
  # Run the Python script to generate embeddings
  python embeddings.py
else
  echo "Using existing ChromaDB from db/chroma"
fi

# Make sure permissions are correct for Docker
chmod -R 755 db/chroma

echo "Database prepared for Docker build"