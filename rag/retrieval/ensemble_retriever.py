from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Set up dense retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Set up sparse retriever (LangChain's BM25Retriever)
# pip install rank-bm25
from langchain_community.retrievers import BM25Retriever as LCBm25Retriever
sparse_retriever = LCBm25Retriever.from_documents(chunks)
sparse_retriever.k = 10

# Combine with configurable weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4]   # 60% semantic, 40% keyword matching
)

results = hybrid_retriever.invoke("Ziegler et al. 2019 RLHF paper")