import time

from prometheus_client import Counter, Gauge, Histogram

from fastapi import Request, Response

retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Stage 1 (BM25 + HNSW + RRF) latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

rerank_latency = Histogram(
    "rag_rerank_latency_seconds",
    "Stage 2 (cross-encoder) latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

llm_tokens_generated = Counter(
    "rag_llm_tokens_generated_total",
    "Total LLM tokens generated across all requests",
)

active_tenants = Gauge(
    "rag_active_tenants_total",
    "Current number of unique tenants with indexed data",
)

http_request_latency = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start

    route = request.url.path
    http_request_latency.labels(
        method=request.method,
        path=route,
        status_code=response.status_code,
    ).observe(elapsed)

    return response
