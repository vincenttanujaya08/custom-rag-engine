#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Custom RAG Engine environment ==="

# Detect platform for GPU acceleration backend
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Detected Apple Silicon – using Metal acceleration"
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
elif command -v nvcc &>/dev/null || [[ -n "${CUDA_HOME:-}" ]]; then
    echo "Detected CUDA toolkit – using CUDA acceleration"
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
else
    echo "No GPU acceleration detected – falling back to CPU"
    pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
fi

# Install remaining Python dependencies
pip install -r requirements.txt

echo "=== Setup complete ==="
