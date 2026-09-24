"""Runtime settings for the capstone pipeline.

Values are plain data with defaults. ``Settings.from_env()`` loads a local
``.env`` file and applies environment overrides, so nothing here touches the
network or the filesystem at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # Source
    sitemap_url: str = "https://docs.langchain.com/sitemap.xml"
    url_filters: tuple[str, ...] = (r"https://docs\.langchain\.com/oss/python/",)

    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_chunk_tokens: int = 50

    # Retrieval
    initial_retrieval_k: int = 20  # candidates before re-ranking
    final_k: int = 5  # chunks passed to the LLM after re-ranking

    # Storage
    persist_directory: Path = Path("./chroma_db")
    collection_name: str = "langchain_docs"
    bm25_index_path: Path = Path("./bm25_index.pkl")

    # Credentials
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from a .env file plus the process environment."""
        load_dotenv()
        return cls(
            persist_directory=Path(os.environ.get("CHROMA_DIR", cls.persist_directory)),
            bm25_index_path=Path(os.environ.get("BM25_INDEX_PATH", cls.bm25_index_path)),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
