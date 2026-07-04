import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest, RAGResponse, IngestRequest, IngestResponse, StreamChunk


def test_query_request_defaults():
    req = QueryRequest(query="hello")
    assert req.query == "hello"
    assert req.top_k == 5
    assert req.threshold == 0.5
    assert req.max_tokens == 512
    assert req.user_id is None
    assert req.use_reranker is False


def test_query_request_with_user():
    req = QueryRequest(query="hello", user_id="alice", use_reranker=True)
    assert req.user_id == "alice"
    assert req.use_reranker is True


def test_query_request_missing_query():
    with pytest.raises(ValidationError):
        QueryRequest()


def test_rag_response():
    resp = RAGResponse(answer="test", sources=[{"text": "src", "score": 0.5}])
    assert resp.answer == "test"
    assert len(resp.sources) == 1


def test_stream_chunk():
    chunk = StreamChunk(token="hello", is_end=False)
    assert chunk.token == "hello"
    assert chunk.is_end is False


def test_ingest_request():
    req = IngestRequest(user_id="bob", documents=["doc1", "doc2"])
    assert req.user_id == "bob"
    assert len(req.documents) == 2


def test_ingest_response():
    resp = IngestResponse(chunks_ingested=5, user_id="bob")
    assert resp.chunks_ingested == 5
    assert resp.user_id == "bob"
