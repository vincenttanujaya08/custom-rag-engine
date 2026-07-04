import math

import numpy as np


def dcg_at_k(relevance: list[float], k: int) -> float:
    top = relevance[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(top))


def ndcg_at_k(relevance: list[float], k: int) -> float:
    dcg_val = dcg_at_k(relevance, k)
    ideal = sorted(relevance, reverse=True)
    idcg_val = dcg_at_k(ideal, k)
    return dcg_val / idcg_val if idcg_val > 0.0 else 0.0


def calculate_ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    relevant_set = set(relevant_ids)
    relevance = [1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids[:k]]
    return ndcg_at_k(relevance, k)
