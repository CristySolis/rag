from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        # Tokenize: lowercase, split by whitespace
        self.tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)
    
    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices by score
        top_k_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        return [self.documents[i] for i in top_k_indices]

# Build BM25 index over your chunks
bm25_retriever = BM25Retriever(chunks)

# Now find Ziegler's paper
results = bm25_retriever.get_relevant_documents("Ziegler et al. 2019", k=5)
# Chunk 1: "Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)..."
# This exact match comes in at rank 1 — exactly where it belongs