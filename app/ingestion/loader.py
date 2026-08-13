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


# ---------------------------------------------------------------------------
# Table helpers (pure Python via pdfplumber — no Java required)
# ---------------------------------------------------------------------------

def _table_to_markdown(table: List[List[Any]]) -> str:
    """
    Converts a pdfplumber table (list of rows, each row is list of cell values)
    into a Markdown table string.
    """
    if not table or not table[0]:
        return ""

    cleaned = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    header = cleaned[0]
    rows = cleaned[1:]
    separator = ["---"] * len(header)

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")

    return "\n".join(lines)


def extract_tables_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts all tables from a PDF using pdfplumber (pure Python, no Java).
    Returns a list of dicts: {"page": int, "table_index": int, "markdown": str}
    """
    if not _HAS_PDFPLUMBER:
        logger.warning("pdfplumber not installed. Skipping table extraction.")
        return []

    extracted_tables = []
    try:
        with _pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    md = _table_to_markdown(table)
                    if md:
                        extracted_tables.append({
                            "page": page_num,
                            "table_index": t_idx,
                            "markdown": md,
                        })
        logger.info(f"Extracted {len(extracted_tables)} tables from {file_path}")
    except Exception as e:
        logger.warning(f"Table extraction failed for {file_path}: {e}")
    return extracted_tables


# ---------------------------------------------------------------------------
# Core text extractors
# ---------------------------------------------------------------------------

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
      1. PyMuPDF — fast, accurate for text-based PDFs.
      2. pdfplumber fallback — handles more encoding variants.
    Scanned/image-only PDFs will still return None (OCR added separately).
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

    logger.warning(f"PyMuPDF returned no text for {file_path}. Trying pdfplumber...")
    fallback = _extract_with_pdfplumber(file_path)
    if fallback:
        logger.info(f"pdfplumber successfully extracted text from {file_path}")
        return fallback

    logger.warning(f"Both extractors failed for {file_path}. PDF may be scanned/image-only.")
    return None


def extract_html_text(file_path: str) -> str:
    """Extracts plain text from HTML files using BeautifulSoup."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for script in soup(["script", "style", "meta", "noscript", "header", "footer"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        logger.error(f"Error reading HTML file {file_path}: {str(e)}")
        raise e


def extract_markdown_text(file_path: str) -> str:
    """Reads markdown file content directly."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading Markdown file {file_path}: {str(e)}")
        raise e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(file_path: str) -> List[Document]:
    """
    Loads a single document file and returns a list of Document objects.
    For PDFs: returns text Document + one Document per extracted table.
    For HTML/Markdown/Text: returns a single Document.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    last_modified = os.path.getmtime(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(filename)[1].lower()

    base_metadata = {
        "filename": filename,
        "file_path": os.path.abspath(file_path),
        "file_size": file_size,
        "last_modified": last_modified,
    }

    documents: List[Document] = []

    if ext == ".pdf" or mime_type == "application/pdf":
        # 1. Main text body
        text = extract_pdf_text(file_path)
        if text:
            documents.append(Document(
                text=text,
                metadata={**base_metadata, "file_type": "pdf", "content_type": "text"}
            ))

        # 2. Tables as individual Documents (each table = 1 searchable chunk)
        tables = extract_tables_from_pdf(file_path)
        for tbl in tables:
            md_text = f"[TABLE -- Page {tbl['page']}]\n\n{tbl['markdown']}"
            documents.append(Document(
                text=md_text,
                metadata={
                    **base_metadata,
                    "file_type": "pdf",
                    "content_type": "table",
                    "table_page": tbl["page"],
                    "table_index": tbl["table_index"],
                }
            ))

        if not documents:
            logger.warning(f"No content extracted from PDF: {filename}")
        return documents

    elif ext in [".html", ".htm"] or mime_type == "text/html":
        text = extract_html_text(file_path)
        if text:
            return [Document(text=text, metadata={**base_metadata, "file_type": "html", "content_type": "text"})]
        return []

    elif ext in [".md", ".markdown"] or mime_type in ["text/markdown", "text/x-markdown"]:
        text = extract_markdown_text(file_path)
        if text:
            return [Document(text=text, metadata={**base_metadata, "file_type": "markdown", "content_type": "text"})]
        return []

    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                return [Document(text=text, metadata={**base_metadata, "file_type": "text", "content_type": "text"})]
        except Exception as e:
            logger.warning(f"Unsupported file type for {filename}: {str(e)}")
        return []


def load_directory(directory_path: str) -> List[Document]:
    """
    Recursively scans a directory for supported documents.
    Returns a flat list of all Document objects (text + tables).
    """
    documents = []
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return documents

    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                docs = load_file(full_path)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error loading {file} from directory: {str(e)}")
                continue

    return documents
