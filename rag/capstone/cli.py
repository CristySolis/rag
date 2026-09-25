"""Command-line entrypoint. Parses arguments, wires settings, prints a summary.

Registered in pyproject.toml as the ``ingest`` script, so after
``poetry install`` it runs as ``poetry run ingest``.
"""
from __future__ import annotations

import argparse
import sys

from rag.capstone.config import Settings
from rag.capstone.ingestion.pipeline import run_ingestion


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Load the LangChain docs, chunk them, and build the dense and sparse indexes.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="only ingest the first N matching pages (useful for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and chunk only; do not build indexes or call the embedding API",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env()

    print(f"Loading docs from {settings.sitemap_url} ...")
    try:
        report = run_ingestion(settings, max_pages=args.max_pages, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {report.pages} pages")
    print(f"Created {report.chunks} chunks (>= {settings.min_chunk_tokens} tokens)")
    if report.vectors is None:
        print("Dry run: indexes not built")
    else:
        print(f"Dense index: {report.vectors} vectors in {settings.persist_directory}")
        print(f"BM25 index saved to {settings.bm25_index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
