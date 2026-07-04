import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import recall_at_k, mean_reciprocal_rank, average_latency
from src.rag.embeddings import RawEmbedder
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.cross_encoder import RawCrossEncoder
from src.rag.two_stage_retriever import TwoStageRetriever
from src.rag.hybrid_fusion import reciprocal_rank_fusion

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)

TOPICS = [
    "renewable energy solar wind power",
    "machine learning neural networks training data",
    "quantum computing superposition entanglement qubits",
    "classical music composers symphonies orchestras",
    "marine biology coral reefs ocean ecosystems",
    "ancient roman history empire civilization",
    "nutrition diet health vitamins minerals",
    "space exploration rockets satellites orbits",
    "python programming software engineering best practices",
    "climate change carbon emissions global warming",
]

VARIATIONS = [
    "Overview and introduction to key concepts.",
    "Historical development and timeline.",
    "Current state-of-the-art research.",
    "Practical applications and use cases.",
    "Challenges and limitations.",
    "Future directions and emerging trends.",
    "Comparison with alternative approaches.",
    "Key contributors and notable works.",
    "Economic impact and market analysis.",
    "Ethical considerations and societal implications.",
]


def build_corpus(num_chunks: int = 100) -> list[str]:
    chunks = []
    for i in range(num_chunks):
        topic = TOPICS[i % len(TOPICS)]
        variation = VARIATIONS[(i // len(TOPICS)) % len(VARIATIONS)]
        chunks.append(f"[{i}] {topic}: {variation}")
    return chunks


GROUND_TRUTH: dict[str, set[int]] = {
    "What are the best ways to generate solar power?": {0, 30, 60, 90},
    "How do neural networks learn from training data?": {11, 31, 51, 71},
    "Explain superposition in quantum computing": {2, 32, 72},
    "Who were the major classical music composers?": {3, 53, 83},
    "What threatens coral reef ecosystems?": {4, 34, 64, 94},
    "How did the Roman Empire expand?": {5, 35, 65},
    "What vitamins are essential for good nutrition?": {6, 36, 56, 76, 96},
    "How do satellites maintain their orbits?": {8, 38, 58, 78},
    "Best practices for Python software engineering": {9, 19, 39, 59, 79},
    "What causes carbon emissions to accelerate?": {0, 10, 30, 50, 70, 90},
}


def main():
    logger.info("Building corpus of 100 chunks ...")
    corpus = build_corpus(100)
    corpus_ids = list(range(len(corpus)))
    logger.info(f"Corpus size: {len(corpus)} chunks")

    logger.info("Loading embedder ...")
    embedder = RawEmbedder()
    chunk_embs = embedder.embed_texts(corpus)

    logger.info("Indexing BM25 and HNSW ...")
    bm25 = RawBM25()
    bm25.index(corpus)

    hnsw_store = HNSWVectorStore(dim=chunk_embs.shape[1], max_elements=len(corpus))
    hnsw_store.add(corpus, chunk_embs)

    cross_encoder = RawCrossEncoder()
    retriever = TwoStageRetriever(
        embedder=embedder,
        bm25=bm25,
        hnsw_store=hnsw_store,
        cross_encoder=cross_encoder,
        recall_k=50,
    )

    hybrid_recalls_5: list[float] = []
    hybrid_recalls_10: list[float] = []
    hybrid_mrrs: list[float] = []
    hybrid_latencies: list[float] = []

    rerank_recalls_5: list[float] = []
    rerank_recalls_10: list[float] = []
    rerank_mrrs: list[float] = []
    rerank_latencies: list[float] = []

    logger.info(f"\n{'='*80}")
    logger.info(f"{'Query':<50} {'Method':<12} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'Lat(ms)':<8}")
    logger.info(f"{'='*80}")

    for query_text, relevant_ids in GROUND_TRUTH.items():
        short_query = query_text[:48]

        t0 = time.perf_counter()
        bm25_hits = bm25.search(query_text, top_k=50)
        query_emb = embedder.embed_texts([query_text])
        hnsw_hits = hnsw_store.search(query_emb.squeeze(0), top_k=50)
        fused = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=50)
        hybrid_latency = (time.perf_counter() - t0) * 1000

        hybrid_ids = [r.get("index", i) for i, r in enumerate(fused[:10])]
        r5 = recall_at_k(hybrid_ids, relevant_ids, 5)
        r10 = recall_at_k(hybrid_ids, relevant_ids, 10)
        mrr = mean_reciprocal_rank(hybrid_ids, relevant_ids)

        hybrid_recalls_5.append(r5)
        hybrid_recalls_10.append(r10)
        hybrid_mrrs.append(mrr)
        hybrid_latencies.append(hybrid_latency)

        short_q = short_query[:44]
        logger.info(
            f"{short_q:<50} {'Hybrid':<12} {r5:<8.3f} {r10:<8.3f} {mrr:<8.3f} {hybrid_latency:<8.1f}"
        )

        t0 = time.perf_counter()
        result = retriever.retrieve(query_text, top_k=10)
        rerank_latency = (time.perf_counter() - t0) * 1000

        rerank_ids = [r.get("chunk_id", i) for i, r in enumerate(result["reranked"][:10])]
        r5_rr = recall_at_k(rerank_ids, relevant_ids, 5)
        r10_rr = recall_at_k(rerank_ids, relevant_ids, 10)
        mrr_rr = mean_reciprocal_rank(rerank_ids, relevant_ids)

        rerank_recalls_5.append(r5_rr)
        rerank_recalls_10.append(r10_rr)
        rerank_mrrs.append(mrr_rr)
        rerank_latencies.append(rerank_latency)

        logger.info(
            f"{'':>50} {'Reranker':<12} {r5_rr:<8.3f} {r10_rr:<8.3f} {mrr_rr:<8.3f} {rerank_latency:<8.1f}"
        )
        logger.info(f"{'-'*80}")

    logger.info(f"\n{'='*80}")
    logger.info(f"{'SUMMARY':^80}")
    logger.info(f"{'='*80}")
    logger.info(f"{'Metric':<30} {'Hybrid Only':<20} {'Hybrid+Reranker':<20}")
    logger.info(f"{'-'*70}")
    logger.info(
        f"{'Avg Recall@5':<30} {average_latency(hybrid_recalls_5):<20.4f} {average_latency(rerank_recalls_5):<20.4f}"
    )
    logger.info(
        f"{'Avg Recall@10':<30} {average_latency(hybrid_recalls_10):<20.4f} {average_latency(rerank_recalls_10):<20.4f}"
    )
    logger.info(
        f"{'Avg MRR':<30} {average_latency(hybrid_mrrs):<20.4f} {average_latency(rerank_mrrs):<20.4f}"
    )
    logger.info(
        f"{'Avg Latency (ms)':<30} {average_latency(hybrid_latencies):<20.1f} {average_latency(rerank_latencies):<20.1f}"
    )
    logger.info(f"{'='*80}")

    mrr_hybrid = average_latency(hybrid_mrrs)
    mrr_rerank = average_latency(rerank_mrrs)
    if mrr_rerank >= mrr_hybrid:
        logger.info(f"\nRESULT: Reranker improves MRR ({mrr_rerank:.4f} vs {mrr_hybrid:.4f})")
    else:
        logger.info(f"\nRESULT: Reranker MRR ({mrr_rerank:.4f}) vs hybrid ({mrr_hybrid:.4f})")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
