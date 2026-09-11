import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Initialize client
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

# Create an index (one-time setup)
index_name = "product-docs"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,        # must match your embedding model!
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index '{index_name}'")

# Connect to the index
index = pc.Index(index_name)
print(f"Index stats: {index.describe_index_stats()}")

## Indexing documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store connected to Pinecone
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"    # metadata key that stores the text content
)

# Index documents
vectorstore.add_documents(chunks)

# Check status
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")

## Querying with namespace support

# Index customer A's documents in their own namespace
customer_a_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="customer_a"
)
customer_a_store.add_documents(customer_a_chunks)

# Query only within customer A's namespace
results = customer_a_store.similarity_search(
    "contract renewal terms",
    k=5
)
# Will NEVER return results from customer B's data

## Metadata filtering

# Filter during similarity search
results = vectorstore.similarity_search(
    "authentication setup",
    k=5,
    filter={
        "department": {"$eq": "Engineering"},
        "year": {"$gte": 2024}
    }
)

# Available filter operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin