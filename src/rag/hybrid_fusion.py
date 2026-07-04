import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    hnsw_results: list[dict],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    rrf_scores: dict[int, float] = {}

    for rank, item in enumerate(bm25_results, 1):
        idx = item.get("index", id(item))
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    for rank, item in enumerate(hnsw_results, 1):
        idx = item.get("index", id(item))
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    seen_texts: set[str] = set()
    fused = []
    for idx, score in ranked:
        for r in bm25_results + hnsw_results:
            text = r["text"]
            ridx = r.get("index", id(r))
            if ridx == idx and text not in seen_texts:
                fused.append({"text": text, "score": round(score, 6)})
                seen_texts.add(text)
                break
        if len(fused) >= top_k:
            break

    return fused
