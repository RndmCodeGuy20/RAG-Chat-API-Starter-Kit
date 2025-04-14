#!/bin/bash
set -e

# Prepare the database for inclusion in Docker image
./prepare-db.sh

# Build and start the Docker containers
docker-compose up -d

echo "API is running at http://localhost:8000"
echo "To test it, try: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"How to install rcg?\"}'"