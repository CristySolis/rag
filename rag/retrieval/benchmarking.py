queries = [
    # Semantic query — no exact term overlap with document
    ("semantic", "how do I troubleshoot login problems"),
    # Exact-term query — specific proper noun
    ("exact", "Ziegler et al. 2019 fine-tuning paper"),
]

for query_type, query in queries:
    print(f"\n{'='*60}")
    print(f"Query type: {query_type}")
    print(f"Query: {query}")
    
    dense_results = dense_retriever.invoke(query)
    sparse_results = sparse_retriever.invoke(query)
    hybrid_results = hybrid_retriever.invoke(query)
    
    print(f"\nDense  top-1: {dense_results[0].page_content[:120]}")
    print(f"Sparse top-1: {sparse_results[0].page_content[:120]}")
    print(f"Hybrid top-1: {hybrid_results[0].page_content[:120]}")