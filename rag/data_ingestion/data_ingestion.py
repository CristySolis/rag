from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import SitemapLoader
import re
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader


#PyPDF

loader = PyPDFLoader("contract.pdf")
pages = loader.load()

# Each page is a Document with page_content and metadata
print(pages[0].page_content[:300])
print(pages[0].metadata)
# {'source': 'contract.pdf', 'page': 0}

# PyMuPDF

loader = PyMuPDFLoader("contract.pdf")
pages = loader.load()

# PyMuPDF preserves more structure and layout
print(pages[0].metadata)
# {'source': 'contract.pdf', 'page': 0, 'page_count': 45,
#  'author': 'Legal Team', 'creator': 'Word', 'producer': 'Adobe PDF'}



# Scans

def load_scanned_pdf(filepath: str) -> list[Document]:
    """Load a scanned PDF using OCR."""
    images = convert_from_path(filepath, dpi=300)
    documents = []
    
    for page_num, image in enumerate(images):
        # OCR the image
        text = pytesseract.image_to_string(image, lang='eng')
        
        # Clean up common OCR artifacts
        text = text.replace('\x0c', '')  # form feed characters
        text = ' '.join(text.split())    # normalize whitespace
        
        if text.strip():  # skip blank pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filepath,
                    "page": page_num,
                    "extraction_method": "ocr"
                }
            ))
    
    return documents

pages = load_scanned_pdf("scanned_legal_brief.pdf")


# HTMLs
# Target only the main content areas
loader = WebBaseLoader(
    web_paths=["https://docs.example.com/api-reference"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("content", "main-content", "article-body")
        )
    }
)

docs = loader.load()

# For bulk loading of document sites



loader = SitemapLoader(
    web_path="https://docs.example.com/sitemap.xml",
    filter_urls=["https://docs.example.com/"],  # only docs, not blog
)

docs = loader.load()
print(f"Loaded {len(docs)} pages")


# Critical post-processing for htmls

def clean_html_document(doc):
    text = doc.page_content
    # Remove any leftover HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    doc.page_content = text
    return doc

cleaned_docs = [clean_html_document(d) for d in docs]


# Loading word documents
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("employee_handbook.docx")
docs = loader.load()


# Using headings as metadata for chunking


def load_docx_with_headings(filepath: str) -> list[Document]:
    """Load DOCX preserving heading structure in metadata."""
    docx = DocxDocument(filepath)
    documents = []
    current_heading = "Introduction"
    current_text = []
    
    for para in docx.paragraphs:
        style = para.style.name
        
        if style.startswith('Heading'):
            # Save accumulated text under previous heading
            if current_text:
                documents.append(Document(
                    page_content='\n'.join(current_text),
                    metadata={
                        "source": filepath,
                        "section": current_heading,
                        "heading_level": style
                    }
                ))
            current_heading = para.text
            current_text = []
        else:
            if para.text.strip():
                current_text.append(para.text)
    
    # Don't forget the last section
    if current_text:
        documents.append(Document(
            page_content='\n'.join(current_text),
            metadata={"source": filepath, "section": current_heading}
        ))
    
    return documents

sections = load_docx_with_headings("employee_handbook.docx")
for s in sections[:3]:
    print(f"Section: {s.metadata['section']}")
    print(f"Content preview: {s.page_content[:100]}")
    print()


# Loading plain text and csv

loader = TextLoader("release_notes.txt", encoding="utf-8")
docs = loader.load()

# CSVs


loader = CSVLoader(
    file_path="faq.csv",
    source_column="question",          # which column to use as source metadata
    metadata_columns=["category", "last_updated"]  # columns to keep as metadata
)

docs = loader.load()
# Each row becomes a Document
# page_content = all column values joined
# metadata = {"source": question_text, "category": ..., "last_updated": ...}