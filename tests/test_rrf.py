from src.rag.hybrid_fusion import reciprocal_rank_fusion


def test_rrf_merges_two_lists():
    bm25 = [
        {"text": "doc_a", "score": 0.9, "index": 0},
        {"text": "doc_b", "score": 0.8, "index": 1},
        {"text": "doc_c", "score": 0.7, "index": 2},
    ]
    hnsw = [
        {"text": "doc_b", "score": 0.85, "index": 1},
        {"text": "doc_d", "score": 0.6, "index": 3},
    ]
    fused = reciprocal_rank_fusion(bm25, hnsw, top_k=4, rrf_k=60)
    texts = [r["text"] for r in fused]
    # doc_b appears in both lists => highest RRF score
    assert texts[0] == "doc_b"
    assert len(fused) == 4


def test_rrf_deduplicates():
    bm25 = [
        {"text": "doc_a", "score": 0.9, "index": 0},
        {"text": "doc_a", "score": 0.9, "index": 0},
    ]
    hnsw = []
    fused = reciprocal_rank_fusion(bm25, hnsw, top_k=5, rrf_k=60)
    assert len(fused) == 1


def test_rrf_top_k_limit():
    bm25 = [{"text": f"doc_{i}", "score": 1.0, "index": i} for i in range(10)]
    hnsw = []
    fused = reciprocal_rank_fusion(bm25, hnsw, top_k=3, rrf_k=60)
    assert len(fused) == 3


def test_rrf_includes_index():
    bm25 = [{"text": "doc_a", "score": 0.9, "index": 42}]
    hnsw = []
    fused = reciprocal_rank_fusion(bm25, hnsw, top_k=5, rrf_k=60)
    assert fused[0]["index"] == 42


def test_rrf_empty_results():
    fused = reciprocal_rank_fusion([], [], top_k=5, rrf_k=60)
    assert fused == []


def test_rrf_order_reflects_combined_rank():
    bm25 = [
        {"text": "doc_x", "score": 1.0, "index": 0},
        {"text": "doc_y", "score": 0.5, "index": 1},
    ]
    hnsw = [
        {"text": "doc_z", "score": 0.9, "index": 2},
        {"text": "doc_y", "score": 0.5, "index": 1},
    ]
    fused = reciprocal_rank_fusion(bm25, hnsw, top_k=3, rrf_k=60)
    texts = [r["text"] for r in fused]
    # doc_y appears in both lists -> should be #1, then doc_x and doc_z by rank
    assert texts[0] == "doc_y"
