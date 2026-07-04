import logging
import re

import torch

from src.rag.embeddings import RawEmbedder

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class SemanticChunker:
    def __init__(
        self,
        embedder: RawEmbedder,
        threshold: float = 0.5,
        max_tokens: int = 512,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        embeds = self.embedder.embed_texts(sentences)

        boundaries = [0]
        for i in range(len(sentences) - 1):
            sim = torch.cosine_similarity(embeds[i].unsqueeze(0), embeds[i + 1].unsqueeze(0)).item()
            if sim < self.threshold:
                boundaries.append(i + 1)

        groups = []
        for start, end in zip(boundaries, boundaries[1:] + [len(sentences)]):
            group_sentences = sentences[start:end]
            group_text = " ".join(group_sentences)
            groups.append(group_text)

        result = []
        for group in groups:
            token_count = self._count_tokens(group)
            if token_count <= self.max_tokens:
                result.append(group)
            else:
                result.extend(self._split_by_tokens(group))

        return result

    def _count_tokens(self, text: str) -> int:
        tokens = self.embedder.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _split_by_tokens(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            if sent_tokens > self.max_tokens:
                words = sent.split()
                for i in range(0, len(words), 64):
                    sub = " ".join(words[i:i + 64])
                    chunks.append(sub)
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
