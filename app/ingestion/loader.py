import os
import mimetypes
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from bs4 import BeautifulSoup
import pymupdf
try:
    import pdfplumber as _pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

logger = logging.getLogger("agentic_rag.ingestion.loader")

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]

def _extract_with_pdfplumber(file_path: str) -> Optional[str]:
    """Fallback: extract text using pdfplumber (handles more PDF variants)."""
    if not _HAS_PDFPLUMBER:
        return None
    try:
        with _pdfplumber.open(file_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
        text = "\n".join(pages_text).strip()
        return text if text else None
    except Exception as e:
        logger.warning(f"pdfplumber also failed for {file_path}: {e}")
        return None


def extract_pdf_text(file_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file.
    Strategy:
      1. PyMuPDF (fitz) — fast, accurate for text-based PDFs.
      2. pdfplumber fallback — handles more encoding variants and some
         digitally-created PDFs that PyMuPDF misses.
    Scanned/image-only PDFs will still return None (OCR not included).
    """
    try:
        doc = pymupdf.open(file_path)
        text_content = []
        for page in doc:
            text = page.get_text()
            if text:
                text_content.append(text)
        full_text = "\n".join(text_content).strip()
    except Exception as e:
        logger.error(f"PyMuPDF failed for {file_path}: {e}")
        full_text = ""

    if full_text:
        return full_text

    # --- Fallback: pdfplumber ---
    logger.warning(
        f"PyMuPDF returned no text for {file_path}. Trying pdfplumber fallback..."
    )
    fallback = _extract_with_pdfplumber(file_path)
    if fallback:
        logger.info(f"pdfplumber successfully extracted text from {file_path}")
        return fallback

    logger.warning(
        f"Both extractors returned no text for {file_path}. "
        "PDF is likely image/scanned-only. Skipping."
    )
    return None

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
