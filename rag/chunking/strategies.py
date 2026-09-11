
## FIXED - SIZE CHUNKING
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=512,    # characters, not tokens
    chunk_overlap=50,
    separator="\n"     # try to split at newlines; fall back to anywhere
)

chunks = splitter.split_documents(documents)

## RECURSIVE CHARACTER SPLITTING

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=[
        "\n\n",   # paragraph breaks (try first)
        "\n",     # line breaks
        ". ",     # sentence boundaries
        ", ",     # clause boundaries
        " ",      # word boundaries
        ""        # character boundaries (last resort)
    ]
)

chunks = splitter.split_documents(documents)

## WITH TOKEN-COUNTING FOR LLM ALIGNMENT

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

def token_length(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    return len(encoding.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,           # tokens
    chunk_overlap=40,         # tokens
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)

# Verify actual token counts
token_counts = [token_length(c.page_content) for c in chunks]
print(f"Mean chunk size: {sum(token_counts)/len(token_counts):.0f} tokens")
print(f"Max chunk size: {max(token_counts)} tokens")
print(f"Min chunk size: {min(token_counts)} tokens")

## SEMANTIC CHUNKING

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # split at the top 5% of similarity drops
)

chunks = splitter.split_documents(documents)

# Chunks will vary in size — some might be 200 tokens, others 800
sizes = [len(c.page_content.split()) for c in chunks]
print(f"Chunk sizes (words): min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.0f}")

## PROPOSITION-LEVEL CHUNKING

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

decompose_prompt = ChatPromptTemplate.from_template("""
Decompose the following text into a list of simple, self-contained factual propositions.
Each proposition should:
- Express a single fact
- Be understandable without the surrounding context
- Be a complete sentence

Text:
{text}

Return one proposition per line, no numbering, no bullet points.
""")

def propositionize(documents: list[Document]) -> list[Document]:
    propositions = []
    
    for doc in documents:
        # Only process chunks of reasonable size
        if len(doc.page_content.split()) < 20:
            continue
            
        response = llm.invoke(
            decompose_prompt.format_messages(text=doc.page_content)
        )
        
        for line in response.content.strip().split('\n'):
            line = line.strip()
            if len(line) > 20:  # filter very short lines
                propositions.append(Document(
                    page_content=line,
                    metadata={
                        **doc.metadata,
                        "proposition_source": doc.page_content[:100]
                    }
                ))
    
    return propositions

# First do recursive splitting, then propositionize
base_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
base_chunks = base_splitter.split_documents(documents)

propositions = propositionize(base_chunks)
print(f"Generated {len(propositions)} propositions from {len(base_chunks)} chunks")


# TIPS

def add_context_header(chunk: Document, doc_title: str, section: str) -> Document:
    header = f"Document: {doc_title}\nSection: {section}\n\n"
    chunk.page_content = header + chunk.page_content
    return chunk