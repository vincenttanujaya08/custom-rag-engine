import logging
from typing import Optional

import hnswlib
import numpy as np
import torch

logger = logging.getLogger(__name__)


class HNSWVectorStore:
    def __init__(self, dim: int = 384, max_elements: int = 100000, ef_construction: int = 200, M: int = 16):
        self.dim = dim
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)
        self.chunks: list[str] = []
        self.next_id: int = 0

    def add(self, chunks: list[str], embeddings: torch.Tensor):
        vectors = embeddings.numpy().astype(np.float32)
        ids = np.arange(self.next_id, self.next_id + len(chunks))

        if self.next_id == 0 and len(chunks) < 2:
            self.index.add_items(vectors, ids)
        elif len(chunks) > 0:
            self.index.add_items(vectors, ids)

        self.chunks.extend(chunks)
        self.next_id += len(chunks)
        logger.info(f"HNSW store now has {len(self.chunks)} chunks")

    def search(self, query_embedding: torch.Tensor, top_k: int = 10) -> list[dict]:
        if self.next_id == 0:
            return []

        query_np = query_embedding.numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        labels, distances = self.index.knn_query(query_np, k=min(top_k, self.next_id))
        labels = labels[0]
        distances = distances[0]

        results = []
        for idx, dist in zip(labels, distances):
            score = round(float(1.0 - dist), 6)
            results.append({"text": self.chunks[idx], "score": score, "index": int(idx)})

        return results
