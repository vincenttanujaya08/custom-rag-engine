from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.5
    max_tokens: int = 512


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]


class StreamChunk(BaseModel):
    token: str
    is_end: bool = False
