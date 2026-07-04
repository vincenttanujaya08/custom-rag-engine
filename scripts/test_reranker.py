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
from src.rag.reranker import RawCrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERY = "How to fix a flat tire?"
KEYWORDS = ["punctured", "rubber", "wheel", "spare"]


def build_trap_dataset() -> list[str]:
    chunks = [
        "The best pasta recipes use fresh ingredients and slow-simmered sauces.",
        "Regular exercise improves cardiovascular health and reduces stress levels.",
        "Quantum computing leverages superposition and entanglement for computation.",
        "Replacing a punctured rubber wheel with a spare is a straightforward process. First, locate the jack and spare tire in your vehicle. Then, loosen the lug nuts before lifting the car.",
        "The history of ancient Rome spans over a thousand years of civilization.",
        "Machine learning models require large datasets for effective training.",
        "Solar panels convert sunlight into electricity through photovoltaic cells.",
        "The Amazon rainforest produces approximately 20% of the world's oxygen.",
        "Classical music composition follows strict rules of harmony and structure.",
        "Marine biology studies organisms in oceans and other saltwater environments.",
    ]
    return chunks


def main():
    logger.info("Loading embedder ...")
    embedder = RawEmbedder()

    chunks = build_trap_dataset()
    logger.info(f"Built dataset with {len(chunks)} chunks")

    chunk_embs = embedder.embed_texts(chunks)

    bf_store = PyTorchVectorStore()
    bf_store.add(chunks, chunk_embs)

    bm25 = RawBM25()
    bm25.index(chunks)

    hnsw_store = HNSWVectorStore(dim=chunk_embs.shape[1], max_elements=len(chunks))
    hnsw_store.add(chunks, chunk_embs)

    cross_encoder = RawCrossEncoder()

    logger.info(f"\n=== Query: '{QUERY}' ===\n")

    t0 = time.perf_counter()
    bm25_hits = bm25.search(QUERY, top_k=50)
    query_emb = embedder.embed_texts([QUERY])
    hnsw_hits = hnsw_store.search(query_emb.squeeze(0), top_k=50)
    fused = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=50)
    t_stage1 = time.perf_counter() - t0

    hybrid_rank = None
    for i, r in enumerate(fused, 1):
        if any(kw in r["text"].lower() for kw in KEYWORDS):
            hybrid_rank = i
            break
    logger.info(f"Stage 1 (Hybrid) rank of tire-fix chunk: #{hybrid_rank}")
    logger.info(f"Stage 1 latency: {t_stage1 * 1000:.2f} ms")

    t0 = time.perf_counter()
    chunk_texts = [r["text"] for r in fused]
    reranked = cross_encoder.rerank(QUERY, chunk_texts, top_k=5)
    t_stage2 = time.perf_counter() - t0

    rerank_rank = None
    for i, r in enumerate(reranked, 1):
        if any(kw in r["text"].lower() for kw in KEYWORDS):
            rerank_rank = i
            break
    logger.info(f"Stage 2 (Reranker) rank of tire-fix chunk: #{rerank_rank}")
    logger.info(f"Stage 2 latency: {t_stage2 * 1000:.2f} ms")
    logger.info(f"Total two-stage latency: {(t_stage1 + t_stage2) * 1000:.2f} ms")

    logger.info(f"\n=== Top-5 After Reranker ===")
    for i, r in enumerate(reranked, 1):
        preview = r["text"][:100]
        logger.info(f"  {i}. score={r['rerank_score']:.4f}  {preview}...")

    passed = True
    if rerank_rank is None:
        logger.error("FAIL: Reranker did not find the tire-fix chunk at all")
        passed = False
    elif rerank_rank < hybrid_rank:
        logger.info(f"PASS: Reranker improved rank from #{hybrid_rank} to #{rerank_rank}")
    elif rerank_rank == 1:
        logger.info(f"PASS: Reranker ranked tire-fix chunk at #1")
    else:
        logger.info(f"INFO: Reranker rank #{rerank_rank} vs hybrid #{hybrid_rank}")

    if t_stage2 < 0.5:
        logger.info(f"PASS: Reranker latency ({t_stage2 * 1000:.2f} ms) is acceptable")
    else:
        logger.warning(f"Reranker latency ({t_stage2 * 1000:.2f} ms) is high")

    logger.info(f"\n{'ALL RERANKER CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
