# pip install langchain-postgres psycopg2-binary
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

CONNECTION_STRING = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/mydb"

# Create vector store (creates table automatically if it doesn't exist)
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="product_docs",
    connection=CONNECTION_STRING,
    use_jsonb=True  # store metadata as JSONB — enables fast filtering
)

# Index documents
vectorstore.add_documents(chunks)

## The SQL Superpower: Combine Vector and Relational Queries
# The real advantage of pgvector is that you can combine vector similarity with SQL

# Using raw SQL for complex queries
import psycopg2
import numpy as np

conn = psycopg2.connect(CONNECTION_STRING)
cur = conn.cursor()

# Get the query embedding
query_text = "How do I configure SSO?"
query_embedding = embeddings.embed_query(query_text)

# SQL query: semantic search + date filter in one query
cur.execute("""
    SELECT
        content,
        source,
        page_num,
        1 - (embedding <=> %s::vector) AS similarity
    FROM document_chunks
    WHERE created_at > NOW() - INTERVAL '90 days'
      AND source != 'archived_docs.pdf'
    ORDER BY embedding <=> %s::vector
    LIMIT 5;
""", (query_embedding, query_embedding))

results = cur.fetchall()
for content, source, page, similarity in results:
    print(f"[{similarity:.3f}] {source} p.{page}: {content[:100]}")

# This query does something Pinecone and ChromaDB can't do efficiently: combine
# arbitrary SQL filters with vector search in a single database round-trip,
# with full ACID guarantees and transactional consistency.
