import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import QueryRequest, RAGResponse, IngestRequest, IngestResponse
from src.api.generator import async_generate_stream
from src.inference.llama_engine import LocalLLMEngine
from src.rag.embeddings import RawEmbedder
from src.rag.vector_store import PyTorchVectorStore
from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner
from src.rag.prompt_builder import PromptBuilder
from src.rag.orchestrator import RAGPipeline
from src.rag.bm25_engine import RawBM25
from src.rag.hnsw_store import HNSWVectorStore
from src.rag.cross_encoder import RawCrossEncoder
from src.rag.two_stage_retriever import TwoStageRetriever
from src.persistence.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

engine: LocalLLMEngine = None
rag: RAGPipeline = None
embedder: RawEmbedder = None
vector_store: PyTorchVectorStore = None
bm25: RawBM25 = None
hnsw_store: HNSWVectorStore = None
sqlite_store: SQLiteStore = None
retriever: TwoStageRetriever = None


def _persist_path(name: str) -> str:
    return str(DATA_DIR / name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, rag, embedder, vector_store, bm25, hnsw_store, sqlite_store, retriever

    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(f"No GGUF model found in {models_dir}")
    logger.info(f"Loading LLM from {gguf_files[0]} ...")
    engine = LocalLLMEngine(model_path=gguf_files[0])

    logger.info("Initializing RAG pipeline ...")
    embedder = RawEmbedder()
    vector_store = PyTorchVectorStore()
    token_counter = TokenCounter()
    pruner = ContextPruner(token_counter=token_counter)
    prompt_builder = PromptBuilder(template="chatml")

    bm25 = RawBM25()
    hnsw_store = HNSWVectorStore(dim=384, max_elements=100000)
    cross_encoder = RawCrossEncoder()

    sqlite_store = SQLiteStore(db_path=_persist_path("rag_store.db"))

    hnsw_path = _persist_path("hnsw_index")
    bm25_path = _persist_path("bm25_state")

    if Path(f"{hnsw_path}.bin").exists() and Path(f"{bm25_path}.pkl").exists():
        logger.info("Loading persisted index state ...")
        hnsw_store.load(hnsw_path)
        bm25.load(bm25_path)
        logger.info(f"Restored: HNSW={len(hnsw_store.chunks)} chunks, BM25={len(bm25.chunks)} chunks")
    else:
        logger.info("No persisted state found, starting fresh")

    retriever = TwoStageRetriever(
        embedder=embedder,
        bm25=bm25,
        hnsw_store=hnsw_store,
        cross_encoder=cross_encoder,
    )

    rag = RAGPipeline(
        token_counter=token_counter,
        pruner=pruner,
        prompt_builder=prompt_builder,
        retriever=retriever,
        use_reranker=True,
    )
    logger.info("RAG pipeline ready.")
    yield

    logger.info("Persisting index state before shutdown ...")
    hnsw_store.save(hnsw_path)
    bm25.save(bm25_path)
    logger.info("State persisted.")


app = FastAPI(title="Custom RAG Engine", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/chat")
async def chat_endpoint(req: QueryRequest) -> RAGResponse:
    allowed = None
    if req.user_id:
        indices = sqlite_store.get_chunk_indices_by_user(req.user_id)
        allowed = set(indices) if indices else None

    result = rag.query(
        user_query=req.query,
        top_k=req.top_k,
        max_generation_tokens=req.max_tokens,
        user_id=req.user_id,
        allowed_indices=allowed,
    )
    answer = engine.generate(result["prompt"], max_tokens=req.max_tokens)
    return RAGResponse(answer=answer, sources=result["chunks_used"])


@app.post("/api/v1/stream")
async def stream_endpoint(req: QueryRequest):
    allowed = None
    if req.user_id:
        indices = sqlite_store.get_chunk_indices_by_user(req.user_id)
        allowed = set(indices) if indices else None

    result = rag.query(
        user_query=req.query,
        top_k=req.top_k,
        max_generation_tokens=req.max_tokens,
        user_id=req.user_id,
        allowed_indices=allowed,
    )

    async def event_stream():
        async for token in async_generate_stream(engine, result["prompt"], max_tokens=req.max_tokens):
            yield f"data: {json.dumps({'token': token, 'is_end': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'is_end': True, 'sources': result['chunks_used']})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/ingest")
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    from src.rag.chunker import SemanticChunker
    chunker = SemanticChunker()
    all_chunks: list[str] = []
    for doc in req.documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    if not all_chunks:
        return IngestResponse(chunks_ingested=0, user_id=req.user_id)

    start_index = sqlite_store.get_max_chunk_index() + 1

    chunk_embs = embedder.embed_texts(all_chunks)

    hnsw_store.add(all_chunks, chunk_embs)
    combined = bm25.chunks + all_chunks
    bm25.index(combined)
    vector_store.add(all_chunks, chunk_embs)
    sqlite_store.add_chunks(all_chunks, user_id=req.user_id, start_index=start_index)

    hnsw_store.save(_persist_path("hnsw_index"))
    bm25.save(_persist_path("bm25_state"))

    logger.info(f"Ingested {len(all_chunks)} chunks for user '{req.user_id}'")
    return IngestResponse(chunks_ingested=len(all_chunks), user_id=req.user_id)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "engine": engine.device_info if engine else "not loaded"}


frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
