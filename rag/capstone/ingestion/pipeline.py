"""Orchestrate the ingestion steps. Knows the order; knows nothing about the CLI."""
from __future__ import annotations

from dataclasses import dataclass

from rag.capstone.config import Settings
from rag.capstone.ingestion.chunking import chunk_documents
from rag.capstone.ingestion.indexing import build_dense_index, build_sparse_index
from rag.capstone.ingestion.loading import load_docs


@dataclass(frozen=True)
class IngestionReport:
    pages: int
    chunks: int
    vectors: int | None  # None when indexes were not built (dry run)


def run_ingestion(
    settings: Settings,
    max_pages: int | None = None,
    dry_run: bool = False,
) -> IngestionReport:
    """Load, chunk and index the docs.

    With ``dry_run`` the pipeline stops after chunking, which exercises the
    network and the splitter without calling the embedding API.
    """
    if not dry_run and not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env (see .env.example) "
            "or export it, or use --dry-run to skip the embedding step."
        )

    docs = load_docs(settings.sitemap_url, settings.url_filters, max_pages=max_pages)
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_tokens=settings.min_chunk_tokens,
    )
    if dry_run:
        return IngestionReport(pages=len(docs), chunks=len(chunks), vectors=None)

    # Sparse first: it is free and local, so a failed embedding call still
    # leaves something usable on disk.
    build_sparse_index(chunks, k=settings.initial_retrieval_k, path=settings.bm25_index_path)
    vectorstore = build_dense_index(
        chunks,
        embedding_model=settings.embedding_model,
        persist_directory=settings.persist_directory,
        collection_name=settings.collection_name,
    )
    return IngestionReport(
        pages=len(docs),
        chunks=len(chunks),
        vectors=vectorstore._collection.count(),
    )
