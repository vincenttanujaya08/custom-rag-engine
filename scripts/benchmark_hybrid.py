import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import RawEmbedder
from src.rag.vector_store import PyTorchVectorStore
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.hybrid_fusion import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RARE_KEYWORD = "Project X99 Alpha"


def build_dataset() -> list[str]:
    chunks = []
    for i in range(95):
        chunks.append(f"This is a document about general topic number {i}. It discusses common subjects and everyday matters.")
    chunks.append(f"The {RARE_KEYWORD} initiative is a top-secret research program focusing on quantum-enhanced vector search algorithms.")
    chunks.append(f"Engineers working on {RARE_KEYWORD} reported breakthroughs in approximate nearest neighbor search latency.")
    chunks.append(f"{RARE_KEYWORD} uses a novel hybrid retrieval approach combining sparse and dense representations.")
    chunks.append(f"The budget for {RARE_KEYWORD} was approved for 2026, funding further development of the custom RAG engine.")
    chunks.append(f"Results from {RARE_KEYWORD} show a 40% improvement in recall over dense-only retrieval methods.")
    return chunks


def test_bruteforce(store: PyTorchVectorStore, query_emb: torch.Tensor):
    t0 = time.perf_counter()
    for _ in range(100):
        _ = store.search(query_emb, top_k=50, threshold=0.0)
    elapsed = (time.perf_counter() - t0) / 100 * 1000
    return elapsed


def test_hnsw(store: HNSWVectorStore, query_emb: torch.Tensor):
    t0 = time.perf_counter()
    for _ in range(100):
        _ = store.search(query_emb.squeeze(0), top_k=50)
    elapsed = (time.perf_counter() - t0) / 100 * 1000
    return elapsed


def main():
    logger.info("Loading embedder ...")
    embedder = RawEmbedder()

    chunks = build_dataset()
    logger.info(f"Built dataset with {len(chunks)} chunks")

    bm25 = RawBM25()
    bm25.index(chunks)

    logger.info(f"Embedding {len(chunks)} chunks ...")
    chunk_embs = embedder.embed_texts(chunks)

    logger.info("Initializing brute-force PyTorchVectorStore ...")
    bf_store = PyTorchVectorStore()
    bf_store.add(chunks, chunk_embs)

    logger.info("Initializing HNSWVectorStore ...")
    hnsw_store = HNSWVectorStore(dim=chunk_embs.shape[1])
    hnsw_store.add(chunks, chunk_embs)

    logger.info(f"\n=== Search for: '{RARE_KEYWORD}' ===\n")

    bm25_results = bm25.search(RARE_KEYWORD, top_k=50)
    bm25_rank = next((i + 1 for i, r in enumerate(bm25_results) if RARE_KEYWORD in r["text"]), None)
    logger.info(f"BM25 rank of rare-keyword chunk: #{bm25_rank}")
    assert bm25_rank is not None and bm25_rank <= 5, "FAIL: BM25 should find rare keyword near top"

    query_emb = embedder.embed_texts([RARE_KEYWORD])
    bf_results = bf_store.search(query_emb, top_k=50, threshold=0.0)
    bf_rank = next((i + 1 for i, r in enumerate(bf_results) if RARE_KEYWORD in r["text"]), None)
    logger.info(f"Brute-force dense rank of rare-keyword chunk: #{bf_rank}")

    hnsw_results = hnsw_store.search(query_emb.squeeze(0), top_k=50)
    hnsw_rank = next((i + 1 for i, r in enumerate(hnsw_results) if RARE_KEYWORD in r["text"]), None)
    logger.info(f"HNSW dense rank of rare-keyword chunk: #{hnsw_rank}")

    fused = reciprocal_rank_fusion(bm25_results, hnsw_results, top_k=10)
    fused_rank = next((i + 1 for i, r in enumerate(fused) if RARE_KEYWORD in r["text"]), None)
    logger.info(f"Hybrid RRF rank of rare-keyword chunk: #{fused_rank}")
    assert fused_rank is not None and fused_rank <= 3, "FAIL: Hybrid should rank rare keyword near top"

    bf_latency = test_bruteforce(bf_store, query_emb)
    hnsw_latency = test_hnsw(hnsw_store, query_emb)

    logger.info(f"\n=== Latency Benchmark (avg over 100 queries) ===")
    n_chunks = len(chunks)
    logger.info(f"Brute-force (O({n_chunks}) matmul): {bf_latency:.3f} ms")
    logger.info(f"HNSW (O(log N) ANN):              {hnsw_latency:.3f} ms")
    logger.info(f"Speedup:                 {bf_latency / max(hnsw_latency, 0.001):.1f}x")

    if bf_latency > hnsw_latency:
        logger.info("PASS: HNSW is faster than brute-force")
    else:
        logger.warning(f"HNSW not faster for N={len(chunks)} -- expected for small datasets")

    logger.info(f"\n=== Final RRF Top-5 ===")
    for i, r in enumerate(fused[:5], 1):
        preview = r["text"][:90]
        logger.info(f"  {i}. score={r['score']:.4f}  {preview}...")

    logger.info("\nALL HYBRID BENCHMARK CHECKS PASSED")


if __name__ == "__main__":
    main()
