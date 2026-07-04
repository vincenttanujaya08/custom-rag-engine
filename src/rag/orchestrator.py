import logging

from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner
from src.rag.prompt_builder import PromptBuilder
from src.rag.embeddings import RawEmbedder
from src.rag.vector_store import PyTorchVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: RawEmbedder,
        vector_store: PyTorchVectorStore,
        token_counter: TokenCounter,
        pruner: ContextPruner,
        prompt_builder: PromptBuilder,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.token_counter = token_counter
        self.pruner = pruner
        self.prompt_builder = prompt_builder

    def query(
        self,
        user_query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        max_generation_tokens: int = 512,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer accurately.",
    ) -> dict:
        query_emb = self.embedder.embed_texts([user_query])
        retrieved = self.vector_store.search(query_emb, top_k=top_k, threshold=threshold)

        logger.info(f"Retrieved {len(retrieved)} chunks from vector store")

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
