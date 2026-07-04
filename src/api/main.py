import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import QueryRequest, RAGResponse
from src.api.generator import async_generate_stream
from src.inference.llama_engine import LocalLLMEngine
from src.rag.embeddings import RawEmbedder
from src.rag.vector_store import PyTorchVectorStore
from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner
from src.rag.prompt_builder import PromptBuilder
from src.rag.orchestrator import RAGPipeline

logger = logging.getLogger(__name__)

engine: LocalLLMEngine = None
rag: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, rag
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
    rag = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        token_counter=token_counter,
        pruner=pruner,
        prompt_builder=prompt_builder,
    )
    logger.info("RAG pipeline ready.")
    yield


app = FastAPI(title="Custom RAG Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/chat")
async def chat_endpoint(req: QueryRequest) -> RAGResponse:
    result = rag.query(
        user_query=req.query,
        top_k=req.top_k,
        threshold=req.threshold,
        max_generation_tokens=req.max_tokens,
    )
    answer = engine.generate(result["prompt"], max_tokens=req.max_tokens)
    return RAGResponse(answer=answer, sources=result["chunks_used"])


@app.post("/api/v1/stream")
async def stream_endpoint(req: QueryRequest):
    result = rag.query(
        user_query=req.query,
        top_k=req.top_k,
        threshold=req.threshold,
        max_generation_tokens=req.max_tokens,
    )

    async def event_stream():
        async for token in async_generate_stream(engine, result["prompt"], max_tokens=req.max_tokens):
            yield f"data: {json.dumps({'token': token, 'is_end': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'is_end': True, 'sources': result['chunks_used']})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "engine": engine.device_info if engine else "not loaded"}


frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
