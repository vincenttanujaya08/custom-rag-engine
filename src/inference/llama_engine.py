import asyncio
import logging
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LocalLLMEngine:
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = True,
    ):
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model: Optional[Llama] = None
        self.device_info: str = "unknown"

        self._load_model()

    def _load_model(self) -> None:
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            verbose=True,
        )

        if self.n_gpu_layers == -1:
            try:
                logger.info("Attempting GPU offload (all layers)...")
                kwargs["n_gpu_layers"] = -1
                self._model = Llama(**kwargs)
                self.device_info = _detect_backend()
                logger.info(f"Model loaded on GPU. Backend: {self.device_info}")
                return
            except Exception as e:
                logger.warning(
                    f"GPU initialisation failed: {e}. "
                    "Falling back to CPU (n_gpu_layers=0)."
                )
                self.device_info = "cpu (fallback)"
        else:
            kwargs["n_gpu_layers"] = self.n_gpu_layers

        logger.info("Loading model on CPU ...")
        self._model = Llama(**kwargs)
        if self.device_info == "unknown":
            self.device_info = "cpu"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()

        def _iterate():
            for chunk in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stream=True,
            ):
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text

        for token_text in await loop.run_in_executor(None, lambda: list(_iterate())):
            yield token_text
            await asyncio.sleep(0)


def _detect_backend() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
