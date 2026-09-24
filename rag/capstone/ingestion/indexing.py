"""Persist chunks as a dense (Chroma) index and a sparse (BM25) index."""
from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_dense_index(
    chunks: list[Document],
    embedding_model: str,
    persist_directory: Path,
    collection_name: str,
) -> Chroma:
    """Embed the chunks with OpenAI and store them in a persistent Chroma collection."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=embedding_model),
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def build_sparse_index(chunks: list[Document], k: int, path: Path) -> BM25Retriever:
    """Build a BM25 retriever over the chunks and pickle it to ``path``."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    with path.open("wb") as f:
        pickle.dump(retriever, f)
    return retriever
