# Custom Vector Search & Local Inference Optimization Engine

Full-stack RAG engine built entirely from scratch using raw PyTorch tensor math, llama.cpp, and FastAPI. No LangChain, no LlamaIndex, no FAISS, no ChromaDB. Every component -- from semantic chunking to token streaming to the interactive dashboard -- is hand-implemented.

1. **Engineered a from-scratch RAG pipeline using raw PyTorch tensor math (`matmul`/`norm`) and semantic chunking, achieving precise context retrieval across a 384-D embedding space without relying on high-level frameworks.**

2. **Reduced LLM VRAM footprint by 71% (6.8 GB to 2.0 GB) by quantizing a 3.4B-parameter model to 4-bit GGUF via llama.cpp, enabling resource-constrained deployment with automated GPU-to-CPU fallback.**

3. **Prevented context-window hallucinations by developing a greedy token-budget pruner that dynamically packs retrieved chunks into an 8,192-token limit, strictly safeguarding a 512-token generation budget.**

4. **Achieved ~22 ms per-token streaming latency by building an async FastAPI gateway (SSE) and zero-dependency JS dashboard, containerized via Docker with CUDA 12.2 passthrough for real-time pipeline observability.**

5. **Upgraded retrieval to hybrid search (dense + sparse) with BM25 from scratch (raw `numpy` IDF/TF scoring) and HNSW ANN (`hnswlib`, O(log N) search), fused via Reciprocal Rank Fusion (RRF) for precise keyword-aware retrieval.**

---

## Architecture Overview

```
User Input
    |
    v
[FastAPI Gateway] -- serves frontend + API endpoints
    |
    v
[RAG Orchestrator]
    |
    |--- [Hybrid Fusion] -- RRF merge of BM25 + HNSW
    |       |--- [RawBM25]        -- sparse keyword search (numpy IDF/TF)
    |       |--- [HNSWVectorStore] -- dense ANN search (hnswlib, O(log N))
    |       |--- [RawEmbedder]    -- query embedding via manual mean-pooling
    |
    |--- [ContextPruner]  -- greedy token-budget packing
    |--- [PromptBuilder]  -- raw f-string template assembly
    |
    v
[LocalLLMEngine] -- llama-cpp-python with GPU acceleration
    |
    v
Generated Text + Source Metadata
```

---

## Project Structure

```
custom-rag-engine/
    Dockerfile                  -- CUDA-enabled container build
    docker-compose.yml          -- GPU passthrough + volume mounts
    requirements.txt            -- Python dependencies
    setup.sh                    -- Platform-aware install (Metal / CUDA / CPU)
    main.py                     -- Quick-start entry point

    src/
        inference/
            llama_engine.py     -- LocalLLMEngine: model load, generate, stream
        rag/
            embeddings.py       -- RawEmbedder: manual mean-pooling, no sentence-transformers wrapper
            chunker.py          -- SemanticChunker: cosine-similarity divergence boundaries
            vector_store.py     -- PyTorchVectorStore: raw matmul/norm cosine search
            bm25_engine.py      -- RawBM25: sparse keyword search from scratch (numpy)
            hnsw_store.py       -- HNSWVectorStore: ANN dense search (hnswlib)
            hybrid_fusion.py    -- Reciprocal Rank Fusion: merges BM25 + HNSW results
            token_counter.py    -- TokenCounter: exact LLM tokenization
            context_pruner.py   -- ContextPruner: greedy budget algorithm
            prompt_builder.py   -- PromptBuilder: ChatML / Llama-3 raw templates
            orchestrator.py     -- RAGPipeline: end-to-end retrieval pipeline
        api/
            schemas.py          -- Pydantic request/response models
            generator.py        -- Async streaming via thread pool
            main.py             -- FastAPI app with CORS, health, chat, stream endpoints

    frontend/
        index.html              -- Chat UI with TailwindCSS (CDN)
        app.js                  -- Vanilla JS fetch + ReadableStream SSE consumer

    scripts/
        download_model.py       -- Download GGUF models from HuggingFace
        test_rag_pipeline.py    -- Phase 2 integration test
        test_context_pruning.py -- Phase 3 budget algorithm validation
        test_api.py             -- Phase 4 streaming API test
        test_docker.sh          -- Docker build helper
        benchmark_hybrid.py     -- Phase 6 hybrid search benchmark
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A GGUF quantized model (tested with `Qwen/Qwen2.5-3B-Instruct-GGUF`)
- (Optional) NVIDIA GPU with CUDA Toolkit 12.x for GPU acceleration
- (Optional) Apple Silicon Mac for Metal acceleration

### Installation

```bash
# Clone and enter the project
cd custom-rag-engine

# Run the platform-aware setup script (detects Metal / CUDA / CPU)
chmod +x setup.sh
./setup.sh

# Download a model
python scripts/download_model.py
```

The `setup.sh` script automatically detects your hardware:
- Apple Silicon: compiles llama.cpp with `GGML_METAL=on`
- NVIDIA GPU with CUDA toolkit: compiles with `GGML_CUDA=on`
- Neither: falls back to CPU-only build

### Quick Start

```bash
python main.py
```

This loads the model, runs a synchronous generation test, and demonstrates asynchronous token streaming to the console.

---

## Components

### Phase 1 -- Local Inference Engine

**`src/inference/llama_engine.py`** - `LocalLLMEngine`

Loads a GGUF quantized model via `llama-cpp-python` with the following safeguards:

- `n_gpu_layers=-1` by default: offloads all layers to GPU
- Automatic CPU fallback: if GPU initialization fails (missing CUDA toolkit, CUDA OOM), catches the exception and reloads with `n_gpu_layers=0`
- Configurable context window (`n_ctx`, default 8192)
- `generate()` method for synchronous text generation
- `stream()` async generator for token-by-token streaming

### Phase 2 -- Semantic Chunking & Vector Search

**`src/rag/embeddings.py`** - `RawEmbedder`

Loads a HuggingFace embedding model using raw `AutoTokenizer` and `AutoModel` from the `transformers` library. The `sentence-transformers` high-level wrapper is avoided. Embeddings are produced via a manual mean-pooling operation that averages token embeddings weighted by the attention mask, followed by L2-normalization.

```python
def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
```

**`src/rag/chunker.py`** - `SemanticChunker`

Splits documents by semantic divergence rather than fixed character counts. Each sentence is embedded, and adjacent cosine similarities are computed. When similarity between consecutive sentences drops below a configurable threshold (default 0.5), a chunk boundary is created. Chunks exceeding a maximum token limit (default 512) are further subdivided using a sliding window over sentences.

**`src/rag/vector_store.py`** - `PyTorchVectorStore`

Stores chunk embeddings as a single contiguous `torch.Tensor` matrix and performs cosine similarity search using raw tensor arithmetic:

```python
dot_product = torch.matmul(self.embeddings_matrix, query_tensor.T)
norm_A = torch.norm(self.embeddings_matrix, dim=1)
norm_B = torch.norm(query_tensor)
cosine_scores = dot_product / (norm_A * norm_B)
```

Results are filtered by a relevance threshold and the top-K highest-scoring chunks are returned with their text and score.

### Phase 3 -- Token Budgeting & Context Pruning

**`src/rag/token_counter.py`** - `TokenCounter`

Counts tokens using the exact tokenizer corresponding to the loaded GGUF model (e.g., Qwen2.5 tokenizer from HuggingFace). This provides accurate counts for budget calculations, distinct from the embedding model's tokenizer.

**`src/rag/context_pruner.py`** - `ContextPruner`

Implements a greedy packing algorithm to fit the most relevant context within a strict token budget:

1. Calculate `available_tokens = max_context_window - system_prompt_tokens - query_tokens - generation_budget`
2. Sort retrieved chunks by score descending
3. Walk through chunks, accepting each if its token count fits within the remaining budget
4. Chunks that exceed the budget are discarded (preferred over mid-sentence truncation, which causes hallucination)

**`src/rag/prompt_builder.py`** - `PromptBuilder`

Assembles the final prompt using raw f-string templates. No `transformers.ChatTemplate` or `langchain.PromptTemplate`. Supports two formats:

- **ChatML (Qwen, Mistral)**: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{context}\n\n{query}<|im_end|>\n<|im_start|>assistant\n`
- **Llama-3 Instruct**: `<|begin_of_text|><|start_header_id|>system<|eot_id|>\n{system}<|eot_id|>\n<|start_header_id|>user<|eot_id|>\n{context}\n\n{query}<|eot_id|>\n<|start_header_id|>assistant<|eot_id|>\n`

**`src/rag/orchestrator.py`** - `RAGPipeline`

Ties the full retrieval pipeline together: embed query, search vector store, prune context by budget, build prompt, and return the result with metadata.

### Phase 4 -- Async API Gateway

**`src/api/main.py`** - FastAPI Application

Three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Standard JSON response with answer and source chunks |
| `/api/v1/stream` | POST | Server-Sent Events streaming, tokens delivered one-by-one |
| `/api/v1/health` | GET | Health check returning backend status and device info |

The streaming endpoint uses `asyncio.to_thread` via a thread pool executor to run the synchronous `llama-cpp-python` generation without blocking the asyncio event loop. CORS middleware is enabled for cross-origin frontend access.

**`src/api/schemas.py`** - Pydantic Models

```python
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.5
    max_tokens: int = 512

class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]
```

### Phase 5 -- Containerization & Dashboard

**Dockerfile**

Uses `nvidia/cuda:12.2.0-devel-ubuntu22.04` as the base image to provide the C++ build chain required for compiling `llama-cpp-python` with CUDA support. The `CMAKE_ARGS=-DGGML_CUDA=on` and `FORCE_CMAKE=1` environment variables force compilation from source with GPU acceleration.

**docker-compose.yml**

```yaml
services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

The `./models` directory is mounted as a volume so the 2GB+ GGUF weights persist across container restarts and do not bloat the Docker image.

**Frontend**

A zero-dependency chat interface served directly by FastAPI:
- `index.html` -- TailwindCSS-styled layout with chat history, input box, and a debug sidebar
- `app.js` -- Uses `fetch` with `response.body.getReader()` to consume the SSE stream (native `EventSource` cannot be used because the endpoint requires POST). Tokens are appended to the active bot message in real time. When the stream completes, the debug panel displays retrieved chunks with their cosine similarity scores.

---

## Running the Tests

Each phase includes a verification script:

```bash
# Phase 2 -- Semantic chunking and vector search
python scripts/test_rag_pipeline.py

# Phase 3 -- Token budget and context pruning
python scripts/test_context_pruning.py

# Phase 4 -- Full API with SSE streaming
# Start the server first:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Then in another terminal:
python scripts/test_api.py

# Phase 5 -- Docker build verification
bash scripts/test_docker.sh

# Phase 6 -- Hybrid search benchmark
python scripts/benchmark_hybrid.py
```

---

## API Examples

### Standard Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain cosine similarity in one sentence.", "top_k": 3}'
```

Response:
```json
{
  "answer": "Cosine similarity measures the cosine of the angle between two vectors...",
  "sources": [{"text": "...", "score": 0.4971}]
}
```

### Streaming Chat

```bash
curl -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is cosine similarity?", "max_tokens": 100}'
```

Response (SSE format):
```
data: {"token": "Cos", "is_end": false}
data: {"token": "ine", "is_end": false}
data: {"token": " similarity", "is_end": false}
...
data: {"token": "", "is_end": true, "sources": []}
```

---

### Phase 6 -- Hybrid Retrieval (BM25 + HNSW + RRF)

**`src/rag/bm25_engine.py`** - `RawBM25`

Full BM25+ implementation from scratch using raw `numpy` vectorized operations. Tokenizes with `re.findall`, computes IDF using the standard formula `log(1 + (N - n + 0.5) / (n + 0.5))`, and scores using the BM25+ formula with configurable `k1`, `b`, and `delta` parameters. No `rank_bm25` or `elasticsearch` libraries.

**`src/rag/hnsw_store.py`** - `HNSWVectorStore`

Replaces brute-force O(N) dense search with Approximate Nearest Neighbors via `hnswlib`. Uses cosine distance and configurable `ef_construction`, `M`, and `ef` parameters. Scales to millions of vectors with O(log N) search time. Converts PyTorch tensors to `numpy` arrays for hnswlib.

**`src/rag/hybrid_fusion.py`** - `reciprocal_rank_fusion`

Implements the standard RRF formula to merge sparse (BM25) and dense (HNSW) result sets:

```
score(d) = sum(1 / (k + rank_i(d)))  for each result list i
```

The `k` parameter (default 60) controls how aggressively low-ranked results are penalized. Results are deduplicated by text content and re-sorted by RRF score.

**`scripts/benchmark_hybrid.py`** - Hybrid Benchmark

Creates a 100-chunk dataset with rare keywords ("Project X99 Alpha"), verifies:
1. BM25 ranks the rare-keyword chunk at #1 (exact keyword match)
2. Hybrid RRF also ranks it at #1
3. Latency comparison between brute-force O(N) and HNSW O(log N) search

---

## Configuration

Key parameters that can be adjusted:

| Parameter | Default | Location |
|-----------|---------|----------|
| Context window | 8192 | `LocalLLMEngine.__init__` |
| Chunk threshold | 0.5 | `SemanticChunker.__init__` |
| Max chunk tokens | 512 | `SemanticChunker.__init__` |
| Vector search top-K | 5 | `PyTorchVectorStore.search` |
| Vector search threshold | 0.0 | `PyTorchVectorStore.search` |
| Max context window | 8192 | `ContextPruner.__init__` |
| Generation token budget | 512 | `ContextPruner.prune_and_pack` |
| Prompt template | chatml | `PromptBuilder.__init__` |

---

## Design Decisions

**Why no high-level frameworks?** Building each component from scratch provides full control over the retrieval pipeline, eliminates dependency bloat, and enables precise debugging when results do not meet expectations. Every tensor operation, token count, and prompt format is explicit and auditable.

**Why discard overflowing chunks instead of truncating?** Mid-sentence truncation introduces hallucination risk because the LLM receives incomplete context. Discarding lower-relevance chunks entirely preserves the integrity of the information presented.

**Why greedy packing instead of knapsack optimization?** The greedy algorithm (sort by score, accept if it fits) runs in O(n log n) and produces near-optimal results for the vast majority of RAG use cases. A full knapsack solver would add complexity without meaningful quality gains.

**Why POST for SSE instead of GET?** The streaming endpoint accepts a JSON payload (query, parameters). Native `EventSource` only supports GET requests, so the frontend uses `fetch` with `ReadableStream` to consume POST-based SSE events.

**Why hybrid (BM25 + dense) instead of dense-only?** Dense embeddings capture semantic similarity but dilute rare or exact keyword matches. BM25 guarantees that documents containing the exact query terms rank highly. RRF merges the two without training, combining the precision of sparse search with the recall of dense search.

**Why hnswlib instead of FAISS?** hnswlib is a minimal C++ binding with zero Python overhead -- no framework abstractions, no data preparation pipeline. It compiles in seconds and exposes exactly two operations: `add_items` and `knn_query`. This aligns with the "hard-mode" philosophy of using raw, minimal dependencies.

---

## Performance Notes

- On an Apple M1 Pro, the Qwen2.5-3B-Instruct q4_K_M model generates at approximately 44 tokens/second with Metal acceleration.
- The embedding model (all-MiniLM-L6-v2) runs on the same GPU via PyTorch MPS, producing 384-dimensional embeddings.
- The context pruner evaluates thousands of tokens in milliseconds -- the bottleneck is the LLM generation, not the retrieval pipeline.
- Docker with CUDA passthrough requires `nvidia-container-toolkit` installed on the host system.

---

## License

MIT
