import math

import pytest


@pytest.fixture
def bm25():
    from src.rag.bm25_engine import RawBM25
    return RawBM25(k1=1.5, b=0.75, delta=1.0)


def test_index_and_vocab(bm25):
    chunks = ["the cat sat on the mat", "the dog ran in the park"]
    bm25.index(chunks)

    assert bm25._built is True
    assert len(bm25.chunks) == 2
    assert "cat" in bm25.vocab
    assert "dog" in bm25.vocab


def test_idf_calculation(bm25):
    chunks = ["hello world", "hello foo", "hello bar"]
    n = len(chunks)
    # "world" appears in 1 doc → idf = log(1 + (3-1+0.5)/(1+0.5)) = log(1 + 2.5/1.5)
    expected_world = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))

    bm25.index(chunks)
    assert math.isclose(bm25.idf["world"], expected_world, rel_tol=1e-9)
    # "hello" appears in all 3 docs → idf = log(1 + (3-3+0.5)/(3+0.5)) = log(1 + 0.5/3.5)
    expected_hello = math.log(1 + (n - 3 + 0.5) / (3 + 0.5))
    assert math.isclose(bm25.idf["hello"], expected_hello, rel_tol=1e-9)


def test_search_simple(bm25):
    chunks = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "birds fly high in the sky",
        "park ranger dog training",
    ]
    bm25.index(chunks)

    results = bm25.search("dog park", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]


def test_search_empty_query(bm25):
    bm25.index(["some text"])
    results = bm25.search("", top_k=5)
    assert results == []


def test_search_not_indexed(bm25):
    with pytest.raises(RuntimeError, match="not indexed"):
        bm25.search("hello")


def test_zero_score_excluded(bm25):
    chunks = ["aaa bbb ccc", "ddd eee fff"]
    bm25.index(chunks)
    results = bm25.search("xxx yyy", top_k=10)
    assert len(results) == 0


def test_score_reproducibility(bm25):
    chunks = ["python programming is fun", "java programming is also fun"]
    bm25.index(chunks)
    r1 = bm25.search("python programming", top_k=2)
    r2 = bm25.search("python programming", top_k=2)
    for a, b in zip(r1, r2):
        assert a["score"] == b["score"]


def test_tf_factor(bm25):
    chunks = ["cat cat cat dog", "cat dog"]
    bm25.index(chunks)
    results = bm25.search("cat", top_k=2)
    assert results[0]["index"] == 0
