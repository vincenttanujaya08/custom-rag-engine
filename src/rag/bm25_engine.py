import json
import logging
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class RawBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: list[str] = []
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0
        self.vocab: list[str] = []
        self._built = False

    def index(self, chunks: list[str]):
        self.chunks = list(chunks)
        self.doc_freqs = [Counter(_tokenize(c)) for c in chunks]

        df: dict[str, int] = {}
        total_terms = 0
        for freq in self.doc_freqs:
            total_terms += sum(freq.values())
            for term in freq:
                df[term] = df.get(term, 0) + 1

        n = len(chunks)
        self.avg_dl = total_terms / n if n > 0 else 1.0

        self.idf = {}
        for term, doc_count in df.items():
            self.idf[term] = math.log(1 + (n - doc_count + 0.5) / (doc_count + 0.5))

        self.vocab = list(self.idf.keys())
        self._built = True
        logger.info(f"BM25 indexed {n} chunks, vocab size: {len(self.vocab)}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._built:
            raise RuntimeError("BM25 not indexed. Call index() first.")

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float32)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(term, 0)
                if tf > 0:
                    dl = sum(freq.values())
                    scores[i] += idf_val * (
                        (tf * (self.k1 + 1))
                        / (tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl)))
                        + self.delta
                    )

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            s = scores[idx]
            if s == 0:
                continue
            results.append({"text": self.chunks[idx], "score": round(float(s), 6), "index": int(idx)})

        return results

    def save(self, path: str):
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "chunks": self.chunks,
            "doc_freqs": [dict(f) for f in self.doc_freqs],
            "idf": self.idf,
            "avg_dl": self.avg_dl,
            "vocab": self.vocab,
            "_built": self._built,
        }

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info(f"BM25 saved to {path}.pkl ({len(self.chunks)} chunks)")

    def load(self, path: str):
        with open(f"{path}.pkl", "rb") as f:
            state = pickle.load(f)

        self.k1 = state["k1"]
        self.b = state["b"]
        self.delta = state["delta"]
        self.chunks = state["chunks"]
        self.doc_freqs = [Counter(d) for d in state["doc_freqs"]]
        self.idf = state["idf"]
        self.avg_dl = state["avg_dl"]
        self.vocab = state["vocab"]
        self._built = state["_built"]

        logger.info(f"BM25 loaded from {path}.pkl ({len(self.chunks)} chunks)")
