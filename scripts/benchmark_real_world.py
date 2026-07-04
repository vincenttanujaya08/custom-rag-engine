import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ndcg import ndcg_at_k
from src.evaluation.metrics import recall_at_k, average_latency
from src.rag.embeddings import RawEmbedder
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.cross_encoder import RawCrossEncoder
from src.rag.two_stage_retriever import TwoStageRetriever
from src.rag.hybrid_fusion import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_real_world")
logger.setLevel(logging.INFO)

RECALL_K = 100
EVAL_K = 10
NUM_QUERIES = 50


def load_scifact():
    logger.info("Loading BEIR SciFact dataset ...")
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="train")

    logger.info(f"Corpus: {len(corpus_ds)} docs, Queries: {len(queries_ds)}, Qrels: {len(qrels_ds)}")

    corpus_texts: list[str] = []
    corpus_id_map: dict[int, str] = {}

    for i, doc in enumerate(corpus_ds):
        parts = [doc.get("title", ""), doc.get("text", "")]
        text = ". ".join(p for p in parts if p)
        corpus_texts.append(text)
        corpus_id_map[i] = doc["_id"]

    qrels_map: dict[str, dict[str, int]] = {}
    for qrel in qrels_ds:
        qid = str(qrel["query-id"])
        cid = str(qrel["corpus-id"])
        score = qrel["score"]
        if qid not in qrels_map:
            qrels_map[qid] = {}
        qrels_map[qid][cid] = score

    qid_to_text: dict[str, str] = {}
    for q in queries_ds:
        qid_to_text[q["_id"]] = q.get("text", "")

    query_list: list[tuple[str, str]] = []
    for qid in sorted(qrels_map.keys()):
        text = qid_to_text.get(qid, "")
        if text:
            query_list.append((qid, text))

    logger.info(f"Queries with qrels: {len(query_list)}")
    return corpus_texts, corpus_id_map, qrels_map, query_list


def build_relevance_arrays(
    retrieved: list[dict],
    query_id: str,
    corpus_id_map: dict[int, str],
    qrels_map: dict[str, dict[str, int]],
    k: int = EVAL_K,
) -> tuple[list[float], list[str], set[str]]:
    retrieved_ids: list[str] = []
    relevance: list[float] = []
    for r in retrieved[:k]:
        idx = r.get("chunk_id") or r.get("index")
        if idx is None:
            raise KeyError(f"Result dict missing both 'chunk_id' and 'index': {r}")
        corpus_id = corpus_id_map[idx]
        retrieved_ids.append(corpus_id)
        rel = qrels_map.get(query_id, {}).get(corpus_id, 0)
        relevance.append(float(rel))
    relevant_ids = {
        cid for cid, rel in qrels_map.get(query_id, {}).items() if rel > 0
    }
    return relevance, retrieved_ids, relevant_ids


def run_strategy(
    query_text: str,
    embedder: RawEmbedder,
    bm25: RawBM25,
    hnsw: HNSWVectorStore,
    retriever: TwoStageRetriever,
    strategy: str,
    top_k: int = EVAL_K,
) -> tuple[list[dict], float]:
    t0 = time.perf_counter()

    if strategy == "dense":
        emb = embedder.embed_texts([query_text])
        results = hnsw.search(emb.squeeze(0), top_k=top_k)

    elif strategy == "hybrid":
        bm25_hits = bm25.search(query_text, top_k=RECALL_K)
        emb = embedder.embed_texts([query_text])
        hnsw_hits = hnsw.search(emb.squeeze(0), top_k=RECALL_K)
        results = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=top_k)

    elif strategy == "hybrid+reranker":
        result = retriever.retrieve(query_text, top_k=top_k)
        results = result["reranked"]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms


def evaluate_strategy(
    strategy_name: str,
    query_list: list[tuple[str, str]],
    embedder: RawEmbedder,
    bm25: RawBM25,
    hnsw: HNSWVectorStore,
    retriever: TwoStageRetriever,
    corpus_id_map: dict[int, str],
    qrels_map: dict[str, dict[str, int]],
    count: int = NUM_QUERIES,
) -> dict:
    all_ndcg: list[float] = []
    all_recall: list[float] = []
    all_latency: list[float] = []

    for qid, qtext in query_list[:count]:
        results, latency_ms = run_strategy(
            qtext, embedder, bm25, hnsw, retriever, strategy_name
        )
        all_latency.append(latency_ms)

        relevance, retrieved_ids, relevant_ids = build_relevance_arrays(
            results, qid, corpus_id_map, qrels_map
        )
        all_ndcg.append(ndcg_at_k(relevance, EVAL_K))
        all_recall.append(recall_at_k(retrieved_ids, relevant_ids, EVAL_K))

    return {
        "strategy": strategy_name,
        "ndcg@10": float(np.mean(all_ndcg)),
        "recall@10": float(np.mean(all_recall)),
        "avg_latency_ms": float(np.mean(all_latency)),
    }


def main():
    corpus_texts, corpus_id_map, qrels_map, query_list = load_scifact()

    num_q = min(NUM_QUERIES, len(query_list))
    logger.info(f"Evaluating first {num_q} queries across all strategies ...")
    logger.info(f"Corpus size: {len(corpus_texts)} documents")

    logger.info("Loading embedder ...")
    embedder = RawEmbedder()
    logger.info("Embedding corpus (batch_size=32) ...")
    chunk_embs = embedder.embed_texts(corpus_texts, batch_size=32)
    logger.info(f"Embedded {len(corpus_texts)} documents, dim={chunk_embs.shape[1]}")

    bm25 = RawBM25()
    bm25.index(corpus_texts)

    hnsw = HNSWVectorStore(dim=chunk_embs.shape[1], max_elements=len(corpus_texts))
    hnsw.add(corpus_texts, chunk_embs)

    cross_encoder = RawCrossEncoder()
    retriever = TwoStageRetriever(
        embedder=embedder,
        bm25=bm25,
        hnsw_store=hnsw,
        cross_encoder=cross_encoder,
        recall_k=RECALL_K,
    )

    strategies = [
        "dense",
        "hybrid",
        "hybrid+reranker",
    ]

    results = []
    for strategy in strategies:
        logger.info(f"  Evaluating strategy: {strategy}")
        result = evaluate_strategy(
            strategy, query_list,
            embedder, bm25, hnsw, retriever,
            corpus_id_map, qrels_map,
            count=num_q,
        )
        results.append(result)
        logger.info(
            f"    NDCG@10={result['ndcg@10']:.4f}, "
            f"Recall@10={result['recall@10']:.4f}, "
            f"Latency={result['avg_latency_ms']:.1f}ms"
        )

    print()
    print("| Retrieval Strategy       | NDCG@10 | Recall@10 | Avg Latency (ms) |")
    print("|--------------------------|---------|-----------|------------------|")
    for r in results:
        print(
            f"| {r['strategy']:<23} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['avg_latency_ms']:.1f} ms |"
        )
    print()

    best = max(results, key=lambda x: x["ndcg@10"])
    baseline = results[0]
    logger.info(
        f"Best strategy: {best['strategy']} (NDCG@10={best['ndcg@10']:.4f}, "
        f"Recall@10={best['recall@10']:.4f})"
    )
    logger.info(
        f"Improvement over {baseline['strategy']}: "
        f"NDCG@10 +{((best['ndcg@10'] - baseline['ndcg@10']) / baseline['ndcg@10'] * 100):.1f}%, "
        f"Recall@10 +{((best['recall@10'] - baseline['recall@10']) / baseline['recall@10'] * 100):.1f}%"
    )
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
