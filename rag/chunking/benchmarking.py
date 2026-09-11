from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
query = "What happens to API calls when the rate limit is exceeded?"

strategies = {
    "fixed_size": CharacterTextSplitter(chunk_size=512, chunk_overlap=50),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
    "semantic": SemanticChunker(embeddings, breakpoint_threshold_type="percentile"),
}

results = {}
for name, splitter in strategies.items():
    chunks = splitter.split_documents(documents)
    db = Chroma.from_documents(chunks, embeddings)
    retrieved = db.similarity_search(query, k=3)
    results[name] = retrieved
    print(f"\n{name.upper()} — {len(chunks)} total chunks")
    for i, doc in enumerate(retrieved):
        print(f"  Chunk {i+1}: {doc.page_content[:150]}")