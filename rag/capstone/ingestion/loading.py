"""Fetch documentation pages from a sitemap and reduce them to clean text."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document

# UI chrome that is not documentation content
_CHROME_TAGS = ["button", "nav", "script", "style", "svg"]
# Inline tags that would otherwise split a sentence across lines
_INLINE_TAGS = ["a", "code", "strong", "em", "b", "i", "span"]


def extract_main_text(soup: BeautifulSoup) -> str:
    """Return the main doc body as text, dropping nav/sidebars and blank runs.

    The docs site is built with Mintlify, which puts the page body in an
    element with id ``content-area``. Falls back to ``main`` and then the
    whole page so unexpected layouts still yield something.
    """
    node = soup.find(id="content-area") or soup.find("main") or soup
    for tag in node.find_all(_CHROME_TAGS):
        tag.decompose()
    for tag in node.find_all(_INLINE_TAGS):
        tag.unwrap()
    node.smooth()
    text = node.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def load_docs(
    sitemap_url: str,
    url_filters: tuple[str, ...],
    max_pages: int | None = None,
) -> list[Document]:
    """Load every page in the sitemap whose URL matches one of the filters.

    ``max_pages`` limits the run to the first N matching URLs, which keeps
    smoke tests fast and free of embedding costs.
    """
    limit = {"blocksize": max_pages, "blocknum": 0} if max_pages else {}
    loader = SitemapLoader(
        web_path=sitemap_url,
        filter_urls=list(url_filters),
        parsing_function=extract_main_text,
        **limit,
    )
    return loader.load()
