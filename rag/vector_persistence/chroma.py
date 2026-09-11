from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Option A: In-memory only (lost when process exits)
vectorstore_memory = Chroma(
    embedding_function=embeddings,
    collection_name="docs_memory"
)

# Option B: Persistent to disk (recommended)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="product_docs"
)

# Add documents (also handles embedding)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader("product_docs.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Index everything in one call
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="product_docs"
)

print(f"Indexed {vectorstore._collection.count()} chunks")

## Querying
# Basic similarity search
results = vectorstore.similarity_search(
    "How do I configure SSO?",
    k=5
)

for doc in results:
    print(f"Score: N/A | Source: {doc.metadata.get('source')}")
    print(doc.page_content[:200])
    print()

# Similarity search with relevance scores (0 to 1, higher = more similar)
results_with_scores = vectorstore.similarity_search_with_relevance_scores(
    "How do I configure SSO?",
    k=5
)

for doc, score in results_with_scores:
    print(f"Score: {score:.3f} | {doc.page_content[:100]}")

## Metadata filtering

# Only retrieve from pages in the "Security" section
results = vectorstore.similarity_search(
    "authentication setup",
    k=5,
    filter={"section": "Security"}
)

# Only from specific source files
results = vectorstore.similarity_search(
    "rate limits",
    k=5,
    filter={"source": "api_reference.pdf"}
)

# Numeric comparisons (e.g., recent documents only)
results = vectorstore.similarity_search(
    "deployment",
    k=5,
    filter={"page": {"$gte": 10, "$lte": 50}}  # pages 10-50 only
)

## Loading an existing index

# Load a previously persisted index — no re-embedding needed
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="product_docs"
)

print(f"Loaded index with {vectorstore._collection.count()} chunks")

## Updating the index

# Add new documents to an existing index
new_docs = loader_new.load()
new_chunks = splitter.split_documents(new_docs)
vectorstore.add_documents(new_chunks)

# Delete by metadata filter (e.g., when a document is updated)
vectorstore._collection.delete(
    where={"source": "old_document.pdf"}
)
