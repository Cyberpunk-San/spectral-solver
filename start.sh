#!/bin/bash

echo "🚀 Starting Spectral Solver..."

# Clean up any existing containers
docker-compose down --remove-orphans

# Build and start services
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."

# Wait for Redis
until docker exec spectral-redis redis-cli ping | grep -q "PONG"; do
    sleep 2
done
echo "✅ Redis is ready!"

# Wait for Ollama (with longer timeout)
until docker exec spectral-ollama curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "⏳ Waiting for Ollama to be ready..."
    sleep 5
done
echo "✅ Ollama is ready!"

# Pull model if needed
echo "📥 Checking for Ollama models..."
if ! docker exec spectral-ollama curl -s http://localhost:11434/api/tags | grep -q "llama3"; then
    echo "📦 Pulling llama3 model..."
    docker exec spectral-ollama ollama pull llama3
fi

echo "🎉 All services are ready!"
echo "📊 Dashboard: http://localhost:8501"
echo "🔧 API: http://localhost:8000"