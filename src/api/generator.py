import asyncio
import json
from typing import AsyncGenerator

from src.inference.llama_engine import LocalLLMEngine


async def async_generate_stream(
    engine: LocalLLMEngine,
    prompt: str,
    max_tokens: int = 512,
) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    def _generate():
        for chunk in engine._model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False,
            stream=True,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text

    for token_text in await loop.run_in_executor(None, lambda: list(_generate())):
        yield token_text
        await asyncio.sleep(0)
