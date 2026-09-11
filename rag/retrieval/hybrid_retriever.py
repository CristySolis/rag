from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], 
    k: int = 60
) -> list[Document]:
    """
    Combine multiple ranked lists of documents using RRF.
    Returns documents sorted by combined RRF score (highest first).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            # Use content as a unique identifier
            doc_id = doc.page_content[:100]
            
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            
            scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by combined score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


class HybridRetriever:
    def __init__(self, documents: list[Document], embeddings):
        self.documents = documents
        
        # Build dense (vector) retriever
        self.vectorstore = Chroma.from_documents(documents, embeddings)
        
        # Build sparse (BM25) retriever
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        # Dense retrieval: top 2k results (we'll fuse and take top k)
        dense_results = self.vectorstore.similarity_search(query, k=k*2)
        
        # Sparse retrieval: top 2k results
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_sparse_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k*2]
        sparse_results = [self.documents[i] for i in top_sparse_indices]
        
        # Fuse with RRF
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:k]


# Usage
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
hybrid_retriever = HybridRetriever(chunks, embeddings)

# Test with an exact-term query
results = hybrid_retriever.retrieve("Ziegler et al. 2019 RLHF")
print("Hybrid results for exact citation query:")
for doc in results:
    print(f"  {doc.page_content[:150]}")

# Test with a semantic query
results = hybrid_retriever.retrieve("how do I fix broken login")
print("\nHybrid results for semantic query:")
for doc in results:
    print(f"  {doc.page_content[:150]}")
