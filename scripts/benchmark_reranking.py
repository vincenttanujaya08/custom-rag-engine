import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import RawEmbedder
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.cross_encoder import RawCrossEncoder
from src.rag.two_stage_retriever import TwoStageRetriever
from src.rag.hybrid_fusion import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERIES = [
    "How to fix a flat tire?",
    "What are the benefits of solar energy?",
    "Explain quantum computing basics",
    "How does machine learning work?",
]

KEYWORD_MAP = {
    "How to fix a flat tire?": ["punctured", "rubber", "wheel", "spare", "tire"],
    "What are the benefits of solar energy?": ["solar", "photovoltaic", "sunlight", "electricity"],
    "Explain quantum computing basics": ["quantum", "superposition", "entanglement"],
    "How does machine learning work?": ["machine learning", "training", "datasets"],
}


def build_trap_dataset() -> list[str]:
    return [
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


def main():
    logger.info("Loading embedder ...")
    embedder = RawEmbedder()

    chunks = build_trap_dataset()
    logger.info(f"Built dataset with {len(chunks)} chunks")

    chunk_embs = embedder.embed_texts(chunks)

    bm25 = RawBM25()
    bm25.index(chunks)

    hnsw_store = HNSWVectorStore(dim=chunk_embs.shape[1], max_elements=len(chunks))
    hnsw_store.add(chunks, chunk_embs)

    cross_encoder = RawCrossEncoder()

    retriever = TwoStageRetriever(
        embedder=embedder,
        bm25=bm25,
        hnsw_store=hnsw_store,
        cross_encoder=cross_encoder,
        recall_k=50,
    )

    total_hybrid_latency = 0.0
    total_rerank_latency = 0.0
    total_hybrid_recall = 0
    total_rerank_recall = 0

    for query in QUERIES:
        keywords = KEYWORD_MAP[query]
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"{'='*60}")

        t0 = time.perf_counter()
        bm25_hits = bm25.search(query, top_k=50)
        query_emb = embedder.embed_texts([query])
        hnsw_hits = hnsw_store.search(query_emb.squeeze(0), top_k=50)
        fused = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=50)
        hybrid_latency = (time.perf_counter() - t0) * 1000
        total_hybrid_latency += hybrid_latency

        hybrid_recall_at_5 = 0
        for r in fused[:5]:
            if any(kw in r["text"].lower() for kw in keywords):
                hybrid_recall_at_5 += 1
        total_hybrid_recall += hybrid_recall_at_5
        logger.info(f"  Hybrid recall@5: {hybrid_recall_at_5}/1  ({hybrid_latency:.1f} ms)")

        t0 = time.perf_counter()
        result = retriever.retrieve(query, top_k=5)
        rerank_latency = (time.perf_counter() - t0) * 1000
        total_rerank_latency += rerank_latency

        rerank_recall_at_5 = 0
        for r in result["reranked"]:
            if any(kw in r["text"].lower() for kw in keywords):
                rerank_recall_at_5 += 1
        total_rerank_recall += rerank_recall_at_5
        logger.info(f"  Two-stage recall@5: {rerank_recall_at_5}/1  ({rerank_latency:.1f} ms)")

        logger.info(f"  Top 5 after reranker:")
        for i, r in enumerate(result["reranked"], 1):
            preview = r["text"][:90]
            logger.info(f"    {i}. score={r['score']:.4f}  {preview}...")

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Avg hybrid latency:    {total_hybrid_latency / len(QUERIES):.1f} ms")
    logger.info(f"Avg two-stage latency: {total_rerank_latency / len(QUERIES):.1f} ms")
    logger.info(f"Total hybrid recall@5:    {total_hybrid_recall}/{len(QUERIES)}")
    logger.info(f"Total two-stage recall@5: {total_rerank_recall}/{len(QUERIES)}")

    passed = True
    if total_rerank_recall < total_hybrid_recall:
        logger.error("FAIL: Reranker degraded recall compared to hybrid-only")
        passed = False
    elif total_rerank_recall >= total_hybrid_recall:
        logger.info("PASS: Reranker maintained or improved recall vs hybrid-only")
    else:
        logger.info("PASS: Equivalent recall")

    if total_rerank_latency / len(QUERIES) < 2000:
        logger.info(f"PASS: Two-stage latency is acceptable")
    else:
        logger.warning(f"Two-stage latency is high")

    logger.info(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
