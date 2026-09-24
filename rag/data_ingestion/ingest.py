from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle
import re
import tiktoken
from rag.config import *

def token_count(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def extract_main_text(soup) -> str:
    """Return the main doc body as text, dropping nav/sidebars and blank runs."""
    node = soup.find(id="content-area") or soup.find("main") or soup
    # Drop UI chrome that is not documentation content
    for tag in node.find_all(["button", "nav", "script", "style", "svg"]):
        tag.decompose()
    # Flatten inline tags so links/code do not split sentences across lines
    for tag in node.find_all(["a", "code", "strong", "em", "b", "i", "span"]):
        tag.unwrap()
    node.smooth()
    text = node.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()

def load_langchain_docs() -> list:
    """Load LangChain documentation from sitemap."""
    print("Loading LangChain documentation...")
    
    # LangChain docs moved from python.langchain.com (Docusaurus) to
    # docs.langchain.com (Mintlify). The old sitemap URL now redirects to an
    # HTML page, so SitemapLoader found zero URLs ("Fetching pages: 0it").
    loader = SitemapLoader(
        web_path="https://docs.langchain.com/sitemap.xml",
        filter_urls=[
            r"https://docs\.langchain\.com/oss/python/",
        ],
        # Only get the main content, skip navigation/sidebars.
        # parsing_function must return a str (not a bs4 Tag) in current
        # langchain-community versions.
        parsing_function=extract_main_text,
    )
    
    docs = loader.load()
    print(f"Loaded {len(docs)} documentation pages")
    return docs

def chunk_documents(documents: list) -> list:
    """Split documents with token-aware recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_count,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata for debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i:05d}"
        chunk.metadata['token_count'] = token_count(chunk.page_content)
    
    # Filter very short chunks (headers, nav items that slipped through)
    chunks = [c for c in chunks if token_count(c.page_content) >= 50]
    
    print(f"Created {len(chunks)} chunks (filtered to >= 50 tokens)")
    return chunks

def build_indexes(chunks: list):
    """Build vector store (dense) and BM25 index (sparse)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Dense index
    print(f"Building dense index with {EMBEDDING_MODEL}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    print(f"Dense index: {vectorstore._collection.count()} vectors")
    
    # Sparse index (BM25) — save to disk
    print("Building BM25 sparse index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = INITIAL_RETRIEVAL_K
    
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 index saved to bm25_index.pkl")
    
    return vectorstore, bm25_retriever

if __name__ == "__main__":
    docs = load_langchain_docs()
    chunks = chunk_documents(docs)
    vectorstore, bm25 = build_indexes(chunks)
    print("\nIngestion complete!")