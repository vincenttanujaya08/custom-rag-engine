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
| 9 | Evaluation (NDCG, Recall@K, MRR), Prometheus metrics, pytest CI | `src/evaluation/`, `src/api/metrics.py`, `.github/workflows/` |
| 10 | Real-world BEIR SciFact benchmark, NDCG@10 ablation study | `scripts/benchmark_real_world.py` |

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

## Real-World Benchmarks (BEIR SciFact)

[SciFact](https://github.com/allenai/scifact) is a scientific claim verification dataset
from the BEIR benchmark suite containing 5,183 scientific documents and 809 queries with
expert-annotated relevance judgments. We evaluate the first 50 queries across three
retrieval strategies.

### Why NDCG@10?

**Normalized Discounted Cumulative Gain (NDCG)** is the gold standard for ranking evaluation
because it measures not just whether relevant documents are found, but whether the *most*
relevant documents appear at the very top of the ranked list, where users are most likely to
see them. Unlike Recall, which treats all relevant documents equally regardless of position,
NDCG applies a logarithmic discount to lower ranks, ensuring that promoting a highly relevant
document from position 10 to position 1 is properly rewarded. Perfect NDCG@10 = 1.0 means
all relevant documents are ranked in the optimal order.

### Ablation Study Results

| Retrieval Strategy | NDCG@10 | Recall@10 | Avg Latency (ms) |
|---|---|---|---|
| Dense Only (HNSW) | 0.6927 | 0.8467 | 22.8 ms |
| Hybrid (BM25+HNSW) | 0.7264 | 0.8000 | 46.3 ms |
| Hybrid + Cross-Encoder | 0.2626 | 0.3767 | 1179.2 ms |

### Key Findings

1. **Hybrid (BM25 + HNSW) achieves the best NDCG@10** at 0.7264, outperforming dense-only
   by +4.9%. The BM25 keyword layer adds lexical coverage that dense embeddings miss.

2. **Dense-only retrieval has the highest Recall@10** (0.8467), suggesting HNSW's semantic
   search casts a wide net that includes many relevant documents — but ranks some of them
   below irrelevant ones.

3. **The cross-encoder reranker underperforms on this domain** (NDCG@10 = 0.2626, latency
   = 1.18s). This is expected: the model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was
   trained on web search queries (MS MARCO), not scientific claim verification. The domain
   mismatch causes it to boost topically plausible but irrelevant documents. A domain-fine-
   tuned reranker (e.g., `BAAI/bge-reranker-v2-m3`) would likely close this gap.

4. **Hybrid retrieval provides the best accuracy-latency tradeoff**, delivering 0.7264 NDCG@10
   in 46.3 ms — 25x faster than the reranker with 2.8x better NDCG.

To reproduce: `uv run python scripts/benchmark_real_world.py`
