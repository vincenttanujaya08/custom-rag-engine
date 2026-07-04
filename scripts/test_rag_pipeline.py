import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import RawEmbedder
from src.rag.chunker import SemanticChunker
from src.rag.vector_store import PyTorchVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEMO_TEXT = """
Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at atomic and subatomic scales. It introduces concepts like wave-particle duality and quantization of energy. The famous Schrodinger's cat thought experiment illustrates the principle of superposition. Quantum entanglement allows particles to be correlated across vast distances.

Baking bread is both an art and a science. The basic ingredients are flour, water, yeast, and salt. The process involves mixing, kneading, fermenting, shaping, and baking. Sourdough bread uses naturally occurring wild yeast and bacteria to leaven the dough. The Maillard reaction gives bread its characteristic golden-brown crust.

The Roman Empire was one of the largest empires in ancient history. It began as a republic and later transitioned into an imperial system under Augustus. Roman engineering achievements include aqueducts, roads, and concrete. The empire's legal and political systems have influenced Western civilization for centuries.
"""


def main():
    logger.info("Initializing RawEmbedder ...")
    embedder = RawEmbedder()

    logger.info("Initializing SemanticChunker ...")
    chunker = SemanticChunker(embedder, threshold=0.5)

    logger.info("Chunking multi-topic document ...")
    chunks = chunker.chunk(DEMO_TEXT)
    logger.info(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        preview = chunk[:100].replace("\n", " ")
        logger.info(f"  Chunk {i + 1}: {preview}...")

    logger.info("Embedding chunks and adding to vector store ...")
    embeddings = embedder.embed_texts(chunks)
    store = PyTorchVectorStore()
    store.add(chunks, embeddings)

    query = "How do you bake a cake?"
    logger.info(f'Query: "{query}"')
    query_emb = embedder.embed_texts([query])
    results = store.search(query_emb, top_k=3, threshold=0.3)

    logger.info(f"Search returned {len(results)} results:")
    for r in results:
        preview = r["text"][:120].replace("\n", " ")
        logger.info(f'  score={r["score"]:.4f}  {preview}...')

    if results:
        top_text = results[0]["text"].lower()
        if any(kw in top_text for kw in ("bread", "baking", "flour", "yeast")):
            logger.info("PASS: Top result is about baking/bread as expected.")
        else:
            logger.warning("Top result may not be about baking - check threshold.")
    else:
        logger.warning("No results returned - threshold may be too high.")

    logger.info("Phase 2 test complete.")


if __name__ == "__main__":
    main()
