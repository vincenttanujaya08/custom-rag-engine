import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, tokenizer=None, model_name: Optional[str] = None):
        self._tokenizer = tokenizer
        self._model_name = model_name or "Qwen/Qwen2.5-3B-Instruct-GGUF"

        if self._tokenizer is None:
            from transformers import AutoTokenizer
            base = self._model_name.replace("-GGUF", "").replace("-Instruct-GGUF", "-Instruct")
            logger.info(f"Loading tokenizer from {base} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(base)
            logger.info("Tokenizer loaded.")

    def count_tokens(self, text: str) -> int:
        if hasattr(self._tokenizer, "tokenize"):
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        return len(self._tokenizer(text)["input_ids"])
