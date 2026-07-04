import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, logging

logging.set_verbosity_info()

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_model(
    repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
) -> Path:
    dest = MODELS_DIR / filename
    if dest.exists():
        print(f"Model already exists at {dest}, skipping download.")
        return dest

    print(f"Downloading {repo_id}/{filename} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model saved to {path}")
    return Path(path)


if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "qwen2.5-3b-instruct-q4_k_m.gguf"
    )
    download_model(repo_id=repo_id, filename=filename)
