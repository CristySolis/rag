import pickle
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from rag.capstone.config import Settings



class HybridRerankedRetriever:
    """
    Two-stage retriever:
    Stage 1: Hybrid (dense + BM25) for high recall
    Stage 2: Cross-encoder re-ranking for high precision
    """
    
    def __init__(self, settings: Settings):
        embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        
        # Load dense retriever
        vectorstore = Chroma(
            persist_directory=str(settings.persist_directory),
            embedding_function=embeddings,
            collection_name=settings.collection_name
        )
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.initial_retrieval_k}
        )
        self.final_k=settings.final_k
        
        # Load BM25 retriever
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = settings.initial_retrieval_k
        
        # Combine as ensemble
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        
        # Load re-ranker
        print(f"Loading re-ranker: {settings.reranker_model}")
        self.reranker = CrossEncoder(settings.reranker_model)
    
    def retrieve(self, query: str) -> list[Document]:
        """
        Full retrieval pipeline:
        1. Hybrid search (dense + BM25)
        2. Cross-encoder re-ranking
        3. Return top FINAL_K chunks
        """
        # Stage 1: Hybrid retrieval
        candidates = self.hybrid_retriever.invoke(query)
        
        # Stage 2: Re-rank
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(
            zip(candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [doc for doc, _ in scored_docs[:self.final_k]]