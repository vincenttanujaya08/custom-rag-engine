import logging
from typing import Optional

import torch

from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner
from src.rag.prompt_builder import PromptBuilder
from src.rag.embeddings import RawEmbedder
from src.rag.vector_store import PyTorchVectorStore
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.hybrid_fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: RawEmbedder,
        vector_store: PyTorchVectorStore,
        token_counter: TokenCounter,
        pruner: ContextPruner,
        prompt_builder: PromptBuilder,
        bm25: Optional[RawBM25] = None,
        hnsw_store: Optional[HNSWVectorStore] = None,
        enable_hybrid: bool = False,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.token_counter = token_counter
        self.pruner = pruner
        self.prompt_builder = prompt_builder
        self.bm25 = bm25
        self.hnsw_store = hnsw_store
        self.enable_hybrid = enable_hybrid

    def query(
        self,
        user_query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        max_generation_tokens: int = 512,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer accurately.",
    ) -> dict:
        if self.enable_hybrid and self.bm25 and self.hnsw_store:
            bm25_hits = self.bm25.search(user_query, top_k=top_k * 10)
            query_emb = self.embedder.embed_texts([user_query])
            hnsw_hits = self.hnsw_store.search(query_emb.squeeze(0), top_k=top_k * 10)
            retrieved = reciprocal_rank_fusion(bm25_hits, hnsw_hits, top_k=top_k)
            logger.info(f"Hybrid search: BM25={len(bm25_hits)}, HNSW={len(hnsw_hits)}, fused={len(retrieved)}")
        else:
            query_emb = self.embedder.embed_texts([user_query])
            retrieved = self.vector_store.search(query_emb, top_k=top_k, threshold=threshold)
            logger.info(f"Dense search retrieved {len(retrieved)} chunks")

        pruned = self.pruner.prune_and_pack(
            system_prompt=system_prompt,
            user_query=user_query,
            retrieved_chunks=retrieved,
            max_generation_tokens=max_generation_tokens,
        )

        prompt = self.prompt_builder.build_prompt(
            system_prompt=system_prompt,
            user_query=user_query,
            context_chunks=pruned,
        )

        total_tokens = self.token_counter.count_tokens(prompt)
        logger.info(f"Final prompt token count: {total_tokens}")

        return {
            "prompt": prompt,
            "chunks_used": pruned,
            "total_chunks_retrieved": len(retrieved),
            "total_chunks_after_pruning": len(pruned),
            "prompt_token_count": total_tokens,
        }
