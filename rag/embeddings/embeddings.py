## OPENAI TEXT-EMBEDDING-3-SMALL

from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Embed a single query
query_vector = embeddings.embed_query("How do I configure SSO?")
print(f"Query vector: {len(query_vector)} dimensions")

# Embed a batch of documents (more efficient than one at a time)
texts = ["First document.", "Second document.", "Third document."]
doc_vectors = embeddings.embed_documents(texts)
print(f"Embedded {len(doc_vectors)} documents")

# DIMENSIONALITY REDUCTOIN

# Request 512 dimensions instead of 1536 — 3x smaller, ~95% of quality
embeddings_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)


## OPENAI TEXT-EMBEDDING-3-LARGE

embeddings_large = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024  # Can reduce from default 3072
)

## COHERE EMBED-V3

from langchain_cohere import CohereEmbeddings

# For indexing documents
doc_embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    input_type="search_document"  # optimized for stored content
)

# For embedding queries
query_embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    input_type="search_query"    # optimized for questions
)

# IMPORTANT: Use the right input_type for each context
doc_vectors = doc_embeddings.embed_documents(chunks)
query_vector = query_embeddings.embed_query("How do I configure SSO?")

## BGE-M3 (OPEN SOURCE, RUNS LOCALLY)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

# Detect available hardware
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={
        "normalize_embeddings": True,  # required for cosine similarity
        "batch_size": 32               # process 32 texts at once
    }
)

# Works identically to OpenAI embeddings in LangChain
query_vector = embeddings.embed_query("How do I configure SSO?")
print(f"Vector dimensions: {len(query_vector)}")  # 1024

## all-MiniLM-L6-v2 (Lightweight Baseline)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)