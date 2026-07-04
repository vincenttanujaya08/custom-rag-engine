import logging
import time
from typing import Optional

import torch

from src.rag.cross_encoder import RawCrossEncoder
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.hybrid_fusion import reciprocal_rank_fusion
from src.rag.embeddings import RawEmbedder

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    def __init__(
        self,
        embedder: RawEmbedder,
        bm25: RawBM25,
        hnsw_store: HNSWVectorStore,
        cross_encoder: RawCrossEncoder,
        recall_k: int = 50,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.bm25 = bm25
        self.hnsw_store = hnsw_store
        self.cross_encoder = cross_encoder
        self.recall_k = recall_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        allowed_indices: Optional[set[int]] = None,
    ) -> dict:
        bm25_hits = self.bm25.search(query, top_k=self.recall_k)
        query_emb = self.embedder.embed_texts([query])
        hnsw_hits = self.hnsw_store.search(query_emb.squeeze(0), top_k=self.recall_k)

        if allowed_indices is not None:
            bm25_hits = [h for h in bm25_hits if h["index"] in allowed_indices]
            hnsw_hits = [h for h in hnsw_hits if h["index"] in allowed_indices]
            logger.info(f"Filtered by user: BM25={len(bm25_hits)} eligible, HNSW={len(hnsw_hits)} eligible")

        fused = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=self.recall_k, rrf_k=self.rrf_k)
        logger.info(f"Stage 1 (Hybrid): BM25={len(bm25_hits)}, HNSW={len(hnsw_hits)}, fused={len(fused)}")

        if not fused:
            return {"candidates": [], "reranked": [], "stage1_latency_ms": 0.0, "stage2_latency_ms": 0.0}

        candidate_texts = [r["text"] for r in fused]
        t0 = time.perf_counter()
        scores = self.cross_encoder.predict(query, candidate_texts)
        stage2_latency = (time.perf_counter() - t0) * 1000

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for idx, score in indexed[:top_k]:
            reranked.append({
                "text": fused[idx]["text"],
                "score": round(float(score), 6),
                "original_index": idx,
            })

        return {
            "candidates": fused,
            "reranked": reranked,
            "stage2_latency_ms": round(stage2_latency, 2),
        }
