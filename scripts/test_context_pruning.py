import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."
USER_QUERY = "What is the best way to bake sourdough bread?"
MAX_CONTEXT = 1024
MAX_GEN_TOKENS = 128


def make_mock_chunks() -> list[dict]:
    chunks = []
    for i in range(5):
        score = 0.95 - i * 0.02
        text = "word " * 1000
        chunks.append({"text": text, "score": round(score, 4)})
    for i in range(15):
        score = 0.85 - i * 0.03
        text = "short chunk number " * 14
        chunks.append({"text": text, "score": round(score, 4)})
    return chunks


def main():
    logger.info("Initializing TokenCounter ...")
    counter = TokenCounter()

    logger.info("Initializing ContextPruner (max_context_window=2048) ...")
    pruner = ContextPruner(token_counter=counter, max_context_window=MAX_CONTEXT)

    chunks = make_mock_chunks()

    system_cost = counter.count_tokens(SYSTEM_PROMPT)
    query_cost = counter.count_tokens(USER_QUERY)
    available = MAX_CONTEXT - system_cost - query_cost - MAX_GEN_TOKENS

    logger.info(f"System tokens: {system_cost}")
    logger.info(f"Query tokens: {query_cost}")
    logger.info(f"Max generation tokens: {MAX_GEN_TOKENS}")
    logger.info(f"Available for context: {available}")

    for i, c in enumerate(chunks):
        t = counter.count_tokens(c["text"])
        logger.info(f"Chunk {i+1}: score={c['score']}, tokens={t}")

    pruned = pruner.prune_and_pack(
        system_prompt=SYSTEM_PROMPT,
        user_query=USER_QUERY,
        retrieved_chunks=chunks,
        max_generation_tokens=MAX_GEN_TOKENS,
    )

    total_pruned_tokens = sum(counter.count_tokens(c["text"]) for c in pruned)
    logger.info(f"Pruned {len(pruned)} chunks, total tokens used: {total_pruned_tokens}")

    passed = True
    for c in pruned:
        t = counter.count_tokens(c["text"])
        if t > 200:
            logger.warning(f"Large chunk snuck through: score={c['score']}, tokens={t}")
            passed = False

    if total_pruned_tokens > available:
        logger.error(f"FAIL: Token budget exceeded ({total_pruned_tokens} > {available})")
        passed = False
    else:
        logger.info(f"PASS: Token budget respected ({total_pruned_tokens} <= {available})")

    if any(c["score"] < 0.5 for c in pruned):
        logger.warning("Low-scoring chunks were accepted - check ordering")

    long_accepted = [c for c in pruned if len(c["text"]) > 5000]
    if long_accepted:
        logger.warning(f"FAIL: {len(long_accepted)} long (1000-token) chunks were accepted")
        passed = False
    else:
        logger.info("PASS: All long (1000-token) chunks were correctly pruned")

    logger.info(f"{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
