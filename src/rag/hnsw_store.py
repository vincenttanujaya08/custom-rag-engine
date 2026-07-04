import json
import logging
from pathlib import Path
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

    def search(self, query_embedding: torch.Tensor, top_k: int = 10, user_id: Optional[str] = None) -> list[dict]:
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

    def save(self, path: str):
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = f"{path}.bin"
        meta_path = f"{path}.json"

        self.index.save_index(index_path)

        meta = {
            "chunks": self.chunks,
            "next_id": self.next_id,
            "dim": self.dim,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        logger.info(f"HNSW store saved to {path}.* ({len(self.chunks)} chunks)")

    def load(self, path: str):
        index_path = f"{path}.bin"
        meta_path = f"{path}.json"

        with open(meta_path) as f:
            meta = json.load(f)

        self.dim = meta["dim"]
        self.chunks = meta["chunks"]
        self.next_id = meta["next_id"]

        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(index_path)
        self.index.set_ef(50)

        logger.info(f"HNSW store loaded from {path}.* ({len(self.chunks)} chunks)")
