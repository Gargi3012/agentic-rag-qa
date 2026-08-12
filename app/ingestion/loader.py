import os
import mimetypes
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from bs4 import BeautifulSoup
import pymupdf

logger = logging.getLogger("agentic_rag.ingestion.loader")

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]

def extract_pdf_text(file_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Logs a warning and skips scanned/non-extractable PDFs.
    """
    try:
        doc = pymupdf.open(file_path)
        text_content = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text:
                text_content.append(text)
        
        full_text = "\n".join(text_content).strip()
        if not full_text:
            logger.warning(f"Scanned or non-extractable PDF detected: {file_path}. Skipping.")
            return None
        
        return full_text
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {str(e)}")
        raise e

def extract_html_text(file_path: str) -> str:
    """
    Extracts plain text from HTML files using BeautifulSoup.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "noscript", "header", "footer"]):
                script.decompose()
                
            # Get text and clean up whitespace
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        logger.error(f"Error reading HTML file {file_path}: {str(e)}")
        raise e

def extract_markdown_text(file_path: str) -> str:
    """
    Reads markdown file content directly.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading Markdown file {file_path}: {str(e)}")
        raise e

def load_file(file_path: str) -> Optional[Document]:
    """
    Loads a single document file, extracts its text, and returns a Document object.
    Supports PDF, HTML, and Markdown.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    last_modified = os.path.getmtime(file_path)
    
    # Guess mime type if extension is non-standard
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(filename)[1].lower()

    text = None
    file_type = "unknown"

    if ext == ".pdf" or mime_type == "application/pdf":
        file_type = "pdf"
        text = extract_pdf_text(file_path)
    elif ext in [".html", ".htm"] or mime_type == "text/html":
        file_type = "html"
        text = extract_html_text(file_path)
    elif ext in [".md", ".markdown"] or mime_type in ["text/markdown", "text/x-markdown"]:
        file_type = "markdown"
        text = extract_markdown_text(file_path)
    else:
        # Fallback to plain text reading
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            file_type = "text"
        except Exception as e:
            logger.warning(f"Unsupported file type for {filename}: {str(e)}")
            return None

    if text is None:
        # Gracefully handle skipped/scanned files
        return None

    metadata = {
        "filename": filename,
        "file_path": os.path.abspath(file_path),
        "file_type": file_type,
        "file_size": file_size,
        "last_modified": last_modified
    }

    return Document(text=text, metadata=metadata)

def load_directory(directory_path: str) -> List[Document]:
    """
    Recursively scans a directory for supportable documents.
    """
    documents = []
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return documents

    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                doc = load_file(full_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error loading {file} from directory: {str(e)}")
                continue
                
    return documents
