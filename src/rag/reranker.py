import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class RawCrossEncoder:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.num_labels = 1
        logger.info(f"Loading cross-encoder '{model_name}' on {self.device} ...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.num_labels = self.model.config.num_labels
            logger.info(f"Cross-encoder has {self.num_labels} output label(s)")
        except Exception as e:
            logger.warning(f"Failed to load on {self.device}: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.num_labels = self.model.config.num_labels

        logger.info(f"Cross-encoder loaded on {self.device}")

    @torch.no_grad()
    def rerank(self, query: str, chunks: list[str], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]

        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        if self.num_labels == 2:
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().tolist()
        else:
            scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:top_k]:
            results.append({
                "text": chunks[idx],
                "rerank_score": round(float(score), 6),
                "original_index": idx,
            })

        return results
