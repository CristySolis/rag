from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.capstone.retrieval.retriever import HybridRerankedRetriever
from rag.capstone.config import Settings

SYSTEM_PROMPT = """You are a helpful assistant for the LangChain Python library.
Answer questions based ONLY on the provided documentation excerpts.
Do not use knowledge from your training data.
If the documentation does not contain the answer, say exactly:
"The LangChain documentation doesn't cover this topic in the provided context."

Always cite the source URL when you reference specific documentation.
Be specific and include code examples when the documentation provides them."""

prompt = ChatPromptTemplate.from_template("""
{system}

Documentation excerpts:
{context}

Question: {question}

Answer:""")



class RAGPipeline:
    def __init__(self, settings: Settings, prompt: ChatPromptTemplate = prompt, llm: ChatOpenAI | None = None):
        self.retriever = HybridRerankedRetriever(settings)
        self.chain = prompt | llm | StrOutputParser()
    
    def answer(self, question: str) -> dict:
        """Answer a question and return answer + sources."""
        # Retrieve
        docs = self.retriever.retrieve(question)
        
        # Format context with source URLs
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate
        answer = self.chain.invoke({
            "system": SYSTEM_PROMPT,
            "context": context,
            "question": question
        })
        
        return {
            "answer": answer,
            "sources": [doc.metadata.get('source') for doc in docs],
            "chunk_count": len(docs)
        }

if __name__ == "__main__":
    settings = Settings.from_env()

    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    pipeline = RAGPipeline(settings, prompt=prompt, llm=llm)
    
    test_questions = [
        "How do I create a simple chain in LangChain?",
        "What is the difference between a Chain and an Agent?",
        "How do I add memory to a conversation chain?",
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = pipeline.answer(question)
        print(f"A: {result['answer'][:400]}")
        print(f"Sources: {result['sources'][:2]}")