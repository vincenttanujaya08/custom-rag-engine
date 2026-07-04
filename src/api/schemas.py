from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.5
    max_tokens: int = 512
    user_id: Optional[str] = None
    use_reranker: bool = False


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]


class StreamChunk(BaseModel):
    token: str
    is_end: bool = False


class IngestRequest(BaseModel):
    user_id: str
    documents: list[str]


class IngestResponse(BaseModel):
    chunks_ingested: int
    user_id: str
