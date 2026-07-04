import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PyTorchVectorStore:
    def __init__(self):
        self.embeddings_matrix: Optional[torch.Tensor] = None
        self.chunks: list[str] = []

    def add(self, chunks: list[str], embeddings: torch.Tensor):
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {embeddings.shape}")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) doesn't match embeddings ({embeddings.shape[0]})"
            )

        if self.embeddings_matrix is None:
            self.embeddings_matrix = embeddings.clone()
        else:
            self.embeddings_matrix = torch.cat([self.embeddings_matrix, embeddings], dim=0)
        self.chunks.extend(chunks)

        logger.info(f"Vector store now has {len(self.chunks)} chunks, dim {self.embeddings_matrix.shape[1]}")

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        if self.embeddings_matrix is None or len(self.chunks) == 0:
            return []

        query_vec = query_embedding.squeeze()
        if query_vec.ndim == 1:
            query_vec = query_vec.unsqueeze(0)

        dot_product = torch.matmul(self.embeddings_matrix, query_vec.T).squeeze(-1)
        norm_db = torch.norm(self.embeddings_matrix, dim=1).clamp(min=1e-9)
        norm_q = torch.norm(query_vec, dim=1).clamp(min=1e-9)
        cosine_scores = dot_product / (norm_db * norm_q)

        scores = cosine_scores.cpu().tolist()
        indexed = list(enumerate(scores))
        filtered = [(i, s) for i, s in indexed if s >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        top = filtered[:top_k]

        return [
            {"text": self.chunks[i], "score": round(s, 6)}
            for i, s in top
        ]
