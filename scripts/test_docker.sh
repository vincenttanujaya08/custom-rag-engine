#!/usr/bin/env bash
set -euo pipefail

echo "=== Building Docker image ==="
docker compose build

echo ""
echo "=== Image built successfully ==="
docker images custom-rag-engine_rag-engine --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "=== To run the container: ==="
echo "  docker compose up"
echo ""
echo "=== To run in detached mode: ==="
echo "  docker compose up -d"
echo ""
echo "=== To view logs: ==="
echo "  docker compose logs -f"
echo ""
echo "=== Notes ==="
echo "  - Ensure the ./models directory contains a .gguf file before starting."
echo "  - If you don't have an NVIDIA GPU, remove the deploy block from docker-compose.yml."
