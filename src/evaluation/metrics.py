import numpy as np


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_latency(timings: list[float]) -> float:
    if not timings:
        return 0.0
    return float(np.mean(timings))
