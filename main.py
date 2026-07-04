import asyncio
import logging
from pathlib import Path

from src.inference.llama_engine import LocalLLMEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"


def find_model() -> Path:
    gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF model found in {MODELS_DIR}. "
            "Run `python scripts/download_model.py` first."
        )
    return gguf_files[0]


async def main():
    model_path = find_model()
    logger.info(f"Loading model from {model_path}")

    engine = LocalLLMEngine(model_path=model_path)
    logger.info(f"Device: {engine.device_info}")

    prompt = "Explain cosine similarity in one sentence."

    logger.info("=== Synchronous generation ===")
    result = engine.generate(prompt, max_tokens=100, temperature=0.7)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}\n")

    logger.info("=== Streaming generation ===")
    print("Stream: ", end="", flush=True)
    async for token in engine.stream(prompt, max_tokens=100, temperature=0.7):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
