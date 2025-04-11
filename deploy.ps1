# Exit on error
$ErrorActionPreference = "Stop"

# Prepare the database for inclusion in Docker image
& ./prepare-db.ps1

# Build and start the Docker containers
# docker-compose up -d

docker buildx build `
    --platform linux/amd64 `
    -t rndmcodeguy/rag-chat-api:v1.0.2 `
    --build-arg GOOGLE_API_KEY=$Env:GOOGLE_API_KEY `
    --build-arg GITHUB_TOKEN=$Env:GITHUB_TOKEN `
    --build-arg CHROMA_PATH=$Env:CHROMA_PATH `
    --build-arg DATA_STORE_PATH=$Env:DATA_STORE_PATH `
    --push .

Write-Host "API is running at http://localhost:8000"
Write-Host "To test it, try: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"How to install rcg?\"}'"