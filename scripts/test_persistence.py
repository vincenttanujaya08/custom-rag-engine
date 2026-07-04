import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import RawEmbedder
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.cross_encoder import RawCrossEncoder
from src.rag.two_stage_retriever import TwoStageRetriever
from src.persistence.sqlite_store import SQLiteStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = str(DATA_DIR / "test_rag_store.db")
HNSW_PATH = str(DATA_DIR / "test_hnsw_index")
BM25_PATH = str(DATA_DIR / "test_bm25_state")


def cleanup():
    for p in [f"{DB_PATH}", f"{HNSW_PATH}.bin", f"{HNSW_PATH}.json", f"{BM25_PATH}.pkl"]:
        Path(p).unlink(missing_ok=True)


def main():
    cleanup()
    passed = 0
    total = 0

    embedder = RawEmbedder()

    docs_user1 = [
        "Replacing a punctured rubber wheel with a spare is a straightforward process. First, locate the jack and spare tire in your vehicle. Then, loosen the lug nuts before lifting the car.",
        "The best pasta recipes use fresh ingredients and slow-simmered sauces.",
    ]
    docs_user2 = [
        "Solar panels convert sunlight into electricity through photovoltaic cells.",
        "Quantum computing leverages superposition and entanglement for computation.",
    ]

    chunks_u1 = list(docs_user1)
    chunks_u2 = list(docs_user2)

    logger.info(f"User 1 chunks: {len(chunks_u1)}, User 2 chunks: {len(chunks_u2)}")

    sqlite_store = SQLiteStore(db_path=DB_PATH)
    bm25 = RawBM25()
    hnsw = HNSWVectorStore(dim=384, max_elements=100)

    emb_u1 = embedder.embed_texts(chunks_u1)
    hnsw.add(chunks_u1, emb_u1)
    bm25.index(chunks_u1)
    sqlite_store.add_chunks(chunks_u1, user_id="user_1", start_index=0)

    start_u2 = len(chunks_u1)
    emb_u2 = embedder.embed_texts(chunks_u2)
    hnsw.add(chunks_u2, emb_u2)
    combined = bm25.chunks + chunks_u2
    bm25.index(combined)
    sqlite_store.add_chunks(chunks_u2, user_id="user_2", start_index=start_u2)

    total += 1
    if sqlite_store.get_chunk_count() == len(chunks_u1) + len(chunks_u2):
        logger.info(f"PASS: SQLiteStore has {sqlite_store.get_chunk_count()} chunks")
        passed += 1
    else:
        logger.error(f"FAIL: Expected {len(chunks_u1) + len(chunks_u2)} chunks, got {sqlite_store.get_chunk_count()}")

    total += 1
    u1_indices = sqlite_store.get_chunk_indices_by_user("user_1")
    if u1_indices == list(range(len(chunks_u1))):
        logger.info(f"PASS: User 1 indices correct: {u1_indices}")
        passed += 1
    else:
        logger.error(f"FAIL: User 1 indices: expected {list(range(len(chunks_u1)))}, got {u1_indices}")

    total += 1
    u2_indices = sqlite_store.get_chunk_indices_by_user("user_2")
    if u2_indices == list(range(len(chunks_u1), len(chunks_u1) + len(chunks_u2))):
        logger.info(f"PASS: User 2 indices correct: {u2_indices}")
        passed += 1
    else:
        logger.error(f"FAIL: User 2 indices: expected {list(range(len(chunks_u1), len(chunks_u1) + len(chunks_u2)))}, got {u2_indices}")

    total += 1
    all_user1 = sqlite_store.get_chunks_by_user("user_1")
    if len(all_user1) == len(chunks_u1):
        logger.info(f"PASS: get_chunks_by_user returns {len(all_user1)} chunks for user_1")
        passed += 1
    else:
        logger.error(f"FAIL: Expected {len(chunks_u1)} chunks for user_1, got {len(all_user1)}")

    logger.info("Saving HNSW and BM25 state ...")
    hnsw.save(HNSW_PATH)
    bm25.save(BM25_PATH)

    logger.info("Loading HNSW and BM25 state into new instances ...")
    hnsw2 = HNSWVectorStore(dim=384, max_elements=100)
    hnsw2.load(HNSW_PATH)
    bm25_loaded = RawBM25()
    bm25_loaded.load(BM25_PATH)

    total += 1
    if len(hnsw2.chunks) == len(chunks_u1) + len(chunks_u2):
        logger.info(f"PASS: Loaded HNSW has {len(hnsw2.chunks)} chunks")
        passed += 1
    else:
        logger.error(f"FAIL: Loaded HNSW has {len(hnsw2.chunks)} chunks, expected {len(chunks_u1) + len(chunks_u2)}")

    total += 1
    if hnsw2.next_id == len(chunks_u1) + len(chunks_u2):
        logger.info(f"PASS: Loaded HNSW next_id = {hnsw2.next_id}")
        passed += 1
    else:
        logger.error(f"FAIL: Loaded HNSW next_id = {hnsw2.next_id}, expected {len(chunks_u1) + len(chunks_u2)}")

    total += 1
    if len(bm25_loaded.chunks) == len(chunks_u1) + len(chunks_u2):
        logger.info(f"PASS: Loaded BM25 has {len(bm25_loaded.chunks)} chunks")
        passed += 1
    else:
        logger.error(f"FAIL: Loaded BM25 has {len(bm25_loaded.chunks)} chunks")

    logger.info("Testing search with user isolation ...")
    cross_encoder = RawCrossEncoder()
    retriever = TwoStageRetriever(
        embedder=embedder,
        bm25=bm25_loaded,
        hnsw_store=hnsw2,
        cross_encoder=cross_encoder,
        recall_k=10,
    )

    query = "How to fix a flat tire?"

    total += 1
    result_all = retriever.retrieve(query, top_k=5)
    top_all = [r["text"] for r in result_all["reranked"]]
    has_tire_all = any("punctured" in t.lower() or "spare" in t.lower() for t in top_all)
    logger.info(f"Without user filter, tire chunk in top 5: {has_tire_all}")
    if has_tire_all:
        logger.info(f"PASS: Tire chunk found without user filter")
        passed += 1
    else:
        logger.error(f"FAIL: Tire chunk NOT found without user filter")

    u1_allowed = set(sqlite_store.get_chunk_indices_by_user("user_1"))
    total += 1
    result_u1 = retriever.retrieve(query, top_k=5, allowed_indices=u1_allowed)
    top_u1 = [r["text"] for r in result_u1["reranked"]]
    has_tire_u1 = any("punctured" in t.lower() or "spare" in t.lower() for t in top_u1)
    logger.info(f"User 1 only, tire chunk in top 5: {has_tire_u1}")
    if has_tire_u1:
        logger.info(f"PASS: Tire chunk found for user_1")
        passed += 1
    else:
        logger.error(f"FAIL: Tire chunk NOT found for user_1")

    u2_allowed = set(sqlite_store.get_chunk_indices_by_user("user_2"))
    total += 1
    result_u2 = retriever.retrieve(query, top_k=5, allowed_indices=u2_allowed)
    top_u2 = [r["text"] for r in result_u2["reranked"]]
    has_tire_u2 = any("punctured" in t.lower() or "spare" in t.lower() for t in top_u2)
    logger.info(f"User 2 only, tire chunk in top 5: {has_tire_u2}")
    if not has_tire_u2:
        logger.info(f"PASS: Tire chunk not visible to user_2 (isolation)")
        passed += 1
    else:
        logger.error(f"FAIL: Tire chunk visible to user_2 (isolation broken)")

    total += 1
    solar_query = "solar energy benefits"
    result_u2_solar = retriever.retrieve(solar_query, top_k=5, allowed_indices=u2_allowed)
    top_u2_solar = [r["text"] for r in result_u2_solar["reranked"]]
    has_solar = any("solar" in t.lower() or "photovoltaic" in t.lower() for t in top_u2_solar)
    if has_solar:
        logger.info(f"PASS: Solar chunk found for user_2")
        passed += 1
    else:
        logger.error(f"FAIL: Solar chunk NOT found for user_2")

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        logger.info("ALL PERSISTENCE & TENANCY CHECKS PASSED")
    else:
        logger.error(f"{total - passed} check(s) FAILED")

    cleanup()
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
