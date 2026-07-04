from unittest.mock import MagicMock, patch

import numpy as np  # noqa: F401  pre-load for Python 3.14 compat
import pytest


@pytest.fixture
def mock_llm_engine():
    with patch.dict("sys.modules", {
        "src.inference.llama_engine": MagicMock(),
        "llama_cpp": MagicMock(),
    }):
        yield


@pytest.fixture
def mock_token_counter():
    from src.rag.token_counter import TokenCounter
    mock = MagicMock(spec=TokenCounter)
    mock.count_tokens.return_value = 10
    return mock
