from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Dense retrieval: embed query, find nearest vectors
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

results = retriever.invoke("How do I reset my password?")
# Returns chunks about "account recovery", "forgotten credentials", 
# "login issues" — even if those phrases don't contain "reset" or "password"