import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class RawCrossEncoder:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.batch_size = batch_size

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
    def predict(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []

        all_scores: list[float] = []

        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            pairs = [[query, doc] for doc in batch]

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
                batch_scores = probs[:, 1].cpu().tolist()
            else:
                batch_scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            all_scores.extend(batch_scores)

        return all_scores
