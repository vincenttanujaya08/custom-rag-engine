import logging

from src.rag.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ContextPruner:
    def __init__(self, token_counter: TokenCounter, max_context_window: int = 8192):
        self.token_counter = token_counter
        self.max_context_window = max_context_window

    def prune_and_pack(
        self,
        system_prompt: str,
        user_query: str,
        retrieved_chunks: list[dict],
        max_generation_tokens: int = 512,
    ) -> list[dict]:
        system_cost = self.token_counter.count_tokens(system_prompt)
        query_cost = self.token_counter.count_tokens(user_query)
        available = self.max_context_window - system_cost - query_cost - max_generation_tokens

        logger.info(
            f"Token budget: max={self.max_context_window}, system={system_cost}, "
            f"query={query_cost}, gen={max_generation_tokens}, available={available}"
        )

        if available <= 0:
            logger.warning("No token budget available for context chunks.")
            return []

        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)

        accepted = []
        used = 0
        for chunk in sorted_chunks:
            cost = self.token_counter.count_tokens(chunk["text"])
            if used + cost <= available:
                accepted.append(chunk)
                used += cost
                logger.debug(f"Accepted chunk (score={chunk['score']:.4f}, tokens={cost})")
            else:
                logger.debug(f"Discarded chunk (score={chunk['score']:.4f}, tokens={cost})")

        logger.info(f"Pruner accepted {len(accepted)}/{len(retrieved_chunks)} chunks ({used} tokens used)")
        return accepted
