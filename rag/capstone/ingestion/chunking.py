"""Split documents into token-bounded chunks."""
from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    min_tokens: int,
) -> list[Document]:
    """Recursively split on paragraphs, lines and sentences, measured in tokens.

    Each chunk gets a ``chunk_id`` and ``token_count`` in its metadata. Chunks
    under ``min_tokens`` are dropped; they are almost always headings or nav
    fragments rather than prose.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:05d}"
        chunk.metadata["token_count"] = token_count(chunk.page_content)
    return [c for c in chunks if c.metadata["token_count"] >= min_tokens]
