import logging
import time
from typing import Optional

from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner
from src.rag.prompt_builder import PromptBuilder
from src.rag.two_stage_retriever import TwoStageRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        token_counter: TokenCounter,
        pruner: ContextPruner,
        prompt_builder: PromptBuilder,
        retriever: Optional[TwoStageRetriever] = None,
        use_reranker: bool = False,
    ):
        self.token_counter = token_counter
        self.pruner = pruner
        self.prompt_builder = prompt_builder
        self.retriever = retriever
        self.use_reranker = use_reranker

    def query(
        self,
        user_query: str,
        top_k: int = 5,
        max_generation_tokens: int = 512,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer accurately.",
        user_id: Optional[str] = None,
        allowed_indices: Optional[set[int]] = None,
    ) -> dict:
        retrieval_latency_s = 0.0
        rerank_latency_s = 0.0

        if self.retriever:
            t0 = time.perf_counter()
            result = self.retriever.retrieve(
                user_query,
                top_k=top_k,
                user_id=user_id,
                allowed_indices=allowed_indices,
            )
            stage2_ms = result.get("stage2_latency_ms", 0.0)
            stage1_total = (time.perf_counter() - t0) * 1000
            stage1_ms = stage1_total - stage2_ms

            retrieval_latency_s = stage1_ms / 1000.0
            rerank_latency_s = stage2_ms / 1000.0

            retrieved = result["reranked"] if self.use_reranker else result["candidates"][:top_k]
            logger.info(
                f"Retrieved {len(retrieved)} chunks "
                f"(stage1={stage1_ms:.1f}ms, stage2={stage2_ms:.1f}ms, reranked={self.use_reranker})"
            )
        else:
            retrieved = []
            logger.warning("No retriever configured")

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
            "retrieval_latency_s": retrieval_latency_s,
            "rerank_latency_s": rerank_latency_s,
        }
