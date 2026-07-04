import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class RawEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        logger.info(f"Loading embedding model '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Embedding model loaded. Dim: {self.embedding_dim}")

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = len(texts)

        all_embeddings: list[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = mean_pooling(outputs.last_hidden_state, attention_mask)
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            all_embeddings.append(batch_emb.cpu())

        return torch.cat(all_embeddings, dim=0)
