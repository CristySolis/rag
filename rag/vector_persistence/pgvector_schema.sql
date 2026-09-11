-- In your PostgreSQL instance
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for document chunks
CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),          -- dimension must match embedding model
    source      TEXT,
    page_num    INTEGER,
    section     TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Create an HNSW index for fast similarity search
-- (IVFFlat is the alternative; HNSW is faster for most use cases)
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Performance tuning with HNSW parameters
-- m: number of connections per layer (higher = better recall, more memory)
-- ef_construction: candidates explored at build time (higher = better recall, slower build)
-- ef_search: candidates explored at query time (tune at query time, not index time)

CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 24, ef_construction = 100);

-- Set ef_search at query time for recall/latency tradeoff
SET hnsw.ef_search = 100;  -- default is 40; increase for better recall
