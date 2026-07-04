# Custom RAG Engine

A retrieval-augmented generation engine built from scratch — no LangChain, LlamaIndex,
FAISS, ChromaDB, sentence-transformers wrapper, SQLAlchemy, or Pinecone.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                         │
│  /api/v1/chat  /api/v1/stream  /api/v1/ingest  /health   │
└──────────┬───────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────┐
│                   RAGPipeline (orchestrator)               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │TwoStage    │  │ContextPruner│  │PromptBuilder     │  │
│  │Retriever   │  │(greedy      │  │(ChatML/Llama-3   │  │
│  │            │  │ budget)     │  │ templates)        │  │
│  └─────┬──────┘  └─────────────┘  └──────────────────┘  │
└────────┼──────────────────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────────────────┐
    │               TwoStageRetriever                   │
    │                                                   │
    │  Stage 1 (Hybrid):                                │
    │    ┌──────────┐    ┌────────────┐    ┌────────┐  │
    │    │RawBM25   │    │HNSWVector  │───►│RRF     │  │
    │    │(numpy IDF│    │Store       │    │Fusion  │  │
    │    │ + TF)    │    │(hnswlib)   │    │        │  │
    │    └──────────┘    └────────────┘    └────────┘  │
    │                                                   │
    │  Stage 2 (Cross-Encoder Reranking):               │
    │    ┌─────────────────────────────────────────┐    │
    │    │ RawCrossEncoder (batched predict)        │    │
    │    └─────────────────────────────────────────┘    │
    └───────────────────────────────────────────────────┘
```

## Phases Implemented

| Phase | Feature | Files |
|-------|---------|-------|
| 1 | Local inference engine (GPU, 4-bit GGUF) | `src/inference/llama_engine.py` |
| 2 | Semantic chunking, embedding, brute-force vector search | `src/rag/chunker.py`, `src/rag/embeddings.py`, `src/rag/vector_store.py` |
| 3 | Context pruning, prompt building, orchestration | `src/rag/context_pruner.py`, `src/rag/prompt_builder.py`, `src/rag/token_counter.py` |
| 4 | FastAPI gateway (chat, stream, health) | `src/api/main.py`, `src/api/schemas.py` |
| 5 | Docker, docker-compose, vanilla JS frontend | `Dockerfile`, `docker-compose.yml`, `frontend/` |
| 6 | Hybrid search (BM25 + HNSW + RRF) | `src/rag/bm25_engine.py`, `src/rag/hnsw_store.py`, `src/rag/hybrid_fusion.py` |
| 7 | Cross-encoder reranking (two-stage retrieval) | `src/rag/cross_encoder.py`, `src/rag/two_stage_retriever.py` |
| 8 | Persistence (save/load) + multi-user tenancy | `src/persistence/sqlite_store.py` |

## Key Design Decisions

- **No high-level frameworks**: All math is raw PyTorch (manual mean-pooling, cosine
  similarity via matmul + norm), numpy IDF/TF scoring, raw HuggingFace transformers.
- **Two-stage retrieval**: Hybrid (BM25 + HNSW + RRF) → Cross-encoder reranking.
- **Multi-user isolation**: Chunk indices mapped to `user_id` via SQLite; search results
  filtered post-retrieval to enforce tenancy.
- **Persistence**: HNSW index saved as `.bin` + metadata as `.json`; BM25 state as `.pkl`;
  SQLite for chunk metadata; all persisted on shutdown and on each ingest.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download a GGUF model (e.g., from HuggingFace)
# Place it in models/

# Run the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Ingest documents
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "documents": ["Your text here."]}'

# Query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question?", "user_id": "alice", "use_reranker": true}'
```

## Benchmark Results

| Metric | Hybrid (Stage 1) | Two-Stage (Hybrid + Reranker) |
|--------|:-:|:-:|
| Recall@5 | 4/4 | 4/4 |
| Avg latency | 368 ms | 121 ms |

## Tests

```bash
# Run benchmarks
uv run python scripts/benchmark_hybrid.py
uv run python scripts/benchmark_reranking.py
uv run python scripts/test_persistence.py
```
