import os

from dotenv import load_dotenv

# Load OPENAI_API_KEY etc. from the project's .env file (gitignored)
load_dotenv()

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval
INITIAL_RETRIEVAL_K = 20  # candidates before re-ranking
FINAL_K = 5               # chunks passed to LLM after re-ranking

# Vector store
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "langchain_docs"

# API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")