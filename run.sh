#!/bin/bash

# Detect docker compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
    COMPOSE_UP="$DOCKER_COMPOSE up -d --build"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    COMPOSE_UP="$DOCKER_COMPOSE up -d --build"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
fi

# Start services in detached mode
echo "Starting services..."
eval $COMPOSE_UP

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11436/api/tags > /dev/null 2>&1; do
  echo -n "."
  sleep 2
done
echo " Ready!"

# Check if llama3.2:3b model exists
if ! docker exec ollama ollama list | grep -q "llama3.2:3b"; then
    echo "Downloading llama3.2:3b model (this may take several minutes)..."
    echo "This is a one-time download - subsequent starts will be much faster"
    docker exec ollama ollama pull llama3.2:3b
    
    if [ $? -eq 0 ]; then
        echo "Model llama3.2:3b downloaded successfully!"
    else
        echo "Failed to download model llama3.2:3b"
        exit 1
    fi
else
    echo "Model llama3.2:3b already available"
fi

echo "All services ready!"
echo "- Frontend: http://localhost:7860"
echo "- Ollama API: http://localhost:11436"
echo ""
echo "Press Ctrl+C to stop all services"

# Follow logs
$DOCKER_COMPOSE logs
