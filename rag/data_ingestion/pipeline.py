import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)

def load_document(filepath: str) -> list[Document]:
    """Load any supported document type with error handling."""
    path = Path(filepath)
    extension = path.suffix.lower()
    
    try:
        if extension == '.pdf':
            loader = PyMuPDFLoader(filepath)
        elif extension in ('.docx', '.doc'):
            loader = Docx2txtLoader(filepath)
        elif extension in ('.txt', '.md'):
            loader = TextLoader(filepath, encoding='utf-8')
        elif extension == '.csv':
            loader = CSVLoader(filepath)
        else:
            print(f"Unsupported file type: {extension}")
            return []
        
        docs = loader.load()
        
        # Enrich metadata for all docs from this file
        for doc in docs:
            doc.metadata['filename'] = path.name
            doc.metadata['file_type'] = extension
            doc.metadata['file_size_kb'] = round(path.stat().st_size / 1024, 1)
        
        return docs
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def ingest_directory(directory: str) -> list[Document]:
    """Recursively ingest all supported files in a directory."""
    all_docs = []
    supported = {'.pdf', '.docx', '.txt', '.md', '.csv'}
    
    for filepath in Path(directory).rglob('*'):
        if filepath.suffix.lower() in supported:
            docs = load_document(str(filepath))
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {filepath.name}")
    
    print(f"\nTotal: {len(all_docs)} documents loaded from {directory}")
    return all_docs

# Usage
documents = ingest_directory("./company_docs/")