from unittest.mock import MagicMock

import pytest

from src.rag.token_counter import TokenCounter
from src.rag.context_pruner import ContextPruner


@pytest.fixture
def token_counter():
    mock = MagicMock(spec=TokenCounter)
    # Return token counts based on string length for deterministic testing
    mock.count_tokens.side_effect = lambda text: max(1, len(text) // 10)
    return mock


@pytest.fixture
def pruner(token_counter):
    return ContextPruner(token_counter=token_counter, max_context_window=200)


def test_prune_and_pack_budget_respected(pruner):
    pruner.token_counter.count_tokens.side_effect = lambda text: len(text) // 10
    chunks = [
        {"text": "a" * 50, "score": 0.9},
        {"text": "b" * 50, "score": 0.8},
        {"text": "c" * 50, "score": 0.7},
        {"text": "d" * 50, "score": 0.6},
    ]
    result = pruner.prune_and_pack(
        system_prompt="system",
        user_query="query",
        retrieved_chunks=chunks,
        max_generation_tokens=50,
    )
    assert isinstance(result, list)
    total_cost = sum(
        pruner.token_counter.count_tokens(c["text"]) for c in result
    )
    available = 200 - pruner.token_counter.count_tokens("system") - pruner.token_counter.count_tokens("query") - 50
    assert total_cost <= available, f"Budget exceeded: {total_cost} > {available}"


def test_prune_orders_by_score(pruner):
    chunks = [
        {"text": "low", "score": 0.1},
        {"text": "high", "score": 0.9},
        {"text": "mid", "score": 0.5},
    ]
    result = pruner.prune_and_pack(
        system_prompt="",
        user_query="",
        retrieved_chunks=chunks,
        max_generation_tokens=0,
    )
    scores = [c["score"] for c in result]
    assert scores == sorted(scores, reverse=True), "Chunks not sorted by score descending"


def test_prune_drops_low_scoring_chunks_when_budget_tight(pruner):
    pruner.token_counter.count_tokens.side_effect = lambda text: len(text)
    chunks = [
        {"text": "x" * 180, "score": 0.1},
        {"text": "y" * 10, "score": 0.9},
    ]
    result = pruner.prune_and_pack(
        system_prompt="",
        user_query="",
        retrieved_chunks=chunks,
        max_generation_tokens=50,
    )
    assert len(result) == 1
    assert result[0]["score"] == 0.9


def test_prune_empty_chunks(pruner):
    result = pruner.prune_and_pack(
        system_prompt="system",
        user_query="query",
        retrieved_chunks=[],
        max_generation_tokens=512,
    )
    assert result == []


def test_prune_no_available_budget(pruner):
    pruner.token_counter.count_tokens.side_effect = lambda text: 10000
    chunks = [{"text": "big", "score": 0.9}]
    result = pruner.prune_and_pack(
        system_prompt="system",
        user_query="query",
        retrieved_chunks=chunks,
        max_generation_tokens=0,
    )
    assert result == []
