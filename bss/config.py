import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.embed_model: str = "BAAI/bge-small-en-v1.5"
        self.chunk_size: int = 800
        self.chunk_overlap: int = 120
        self.top_k: int = 4
        self.score_threshold: float = 0.30
        self.llm_model: str = "google/gemma-4-31b-it:free"
        self.llm_fallback: str = "openrouter/free"
        self.openai_base_url: str = "https://openrouter.ai/api/v1"
        self.referer: str = "https://github.com/ARandomDev99/bss"
        self.title: str = "bss-rag"
        self.index_dir: Path = Path("storage")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")


CFG = Config()