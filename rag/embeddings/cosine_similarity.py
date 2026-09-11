import numpy as np

from langchain_openai import OpenAIEmbeddings

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test semantic relationships
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pairs = [
    ("How do I reset my password?", "Steps to recover account access"),
    ("How do I reset my password?", "What are the billing options?"),
    ("cat", "kitten"),
    ("cat", "automobile"),
]

for text1, text2 in pairs:
    v1 = embeddings.embed_query(text1)
    v2 = embeddings.embed_query(text2)
    sim = cosine_similarity(v1, v2)
    print(f"{sim:.3f}: '{text1}' vs '{text2}'")

# Typical output:
# 0.847: 'How do I reset my password?' vs 'Steps to recover account access'
# 0.312: 'How do I reset my password?' vs 'What are the billing options?'
# 0.891: 'cat' vs 'kitten'
# 0.234: 'cat' vs 'automobile'