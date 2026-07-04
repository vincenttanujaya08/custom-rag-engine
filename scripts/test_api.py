import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


def test_health():
    r = httpx.get(f"{BASE_URL}/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    logger.info(f"Health check: {data}")


def test_chat():
    payload = {"query": "Explain cosine similarity in one sentence.", "top_k": 3, "max_tokens": 100}
    r = httpx.post(f"{BASE_URL}/api/v1/chat", json=payload, timeout=120)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    logger.info(f"Chat response: {data['answer'][:80]}...")


def test_stream():
    payload = {"query": "Explain cosine similarity in one sentence.", "top_k": 3, "max_tokens": 100}
    tokens = []
    with httpx.stream("POST", f"{BASE_URL}/api/v1/stream", json=payload, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("is_end"):
                    logger.info(f"Stream complete. Sources: {len(data.get('sources', []))}")
                    break
                tokens.append(data["token"])

    full = "".join(tokens)
    assert len(tokens) > 0, "Should have received at least one token"
    assert len(full) > 10, "Response should be longer than 10 chars"
    logger.info(f"Streamed {len(tokens)} tokens, full text:\n{full}")


if __name__ == "__main__":
    test_health()
    logger.info("=== Testing /api/v1/chat ===")
    test_chat()
    logger.info("=== Testing /api/v1/stream ===")
    test_stream()
    logger.info("ALL API TESTS PASSED")
