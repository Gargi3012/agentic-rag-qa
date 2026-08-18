import re
import logging
import hashlib
import tiktoken
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.ingestion.loader import Document

logger = logging.getLogger("agentic_rag.ingestion.chunker")


class Chunk(BaseModel):
    id: str          # SHA-256 hash set by deduplicator (or computed here)
    text: str
    metadata: Dict[str, Any]
    token_count: int


# ---------------------------------------------------------------------------
# Shared tokenizer utility
# ---------------------------------------------------------------------------

def _get_encoding(encoding_name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        logger.warning(f"Could not load encoding {encoding_name}, falling back to cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def _make_chunk_id(text: str, metadata: Dict[str, Any]) -> str:
    """Deterministic SHA-256 ID from text + source filename + chunk_index."""
    raw = f"{metadata.get('filename', '')}-{metadata.get('chunk_index', 0)}-{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Strategy 1: Table-Aware Chunking (atomic — never split tables)
# ---------------------------------------------------------------------------

def _chunk_table_document(doc: Document) -> List[Chunk]:
    """
    Tables are treated as atomic chunks — never split mid-row.
    Each table Document → exactly 1 Chunk.
    """
    enc = _get_encoding()
    token_count = _count_tokens(doc.text, enc)
    metadata = {**doc.metadata, "chunk_index": 0, "chunking_strategy": "table_atomic"}
    chunk_id = _make_chunk_id(doc.text, metadata)
    return [Chunk(id=chunk_id, text=doc.text, metadata=metadata, token_count=token_count)]


# ---------------------------------------------------------------------------
# Strategy 2: Image-Aware Chunking (atomic — vision description as chunk)
# ---------------------------------------------------------------------------

def _chunk_image_document(doc: Document) -> List[Chunk]:
    """
    Image Documents (with vision-generated text) → exactly 1 Chunk.
    """
    enc = _get_encoding()
    token_count = _count_tokens(doc.text, enc)
    metadata = {**doc.metadata, "chunk_index": 0, "chunking_strategy": "image_atomic"}
    chunk_id = _make_chunk_id(doc.text, metadata)
    return [Chunk(id=chunk_id, text=doc.text, metadata=metadata, token_count=token_count)]


# ---------------------------------------------------------------------------
# Strategy 3: Section-Aware Chunking (heading-based splitting)
# ---------------------------------------------------------------------------

# Heading patterns: Markdown headings or ALL-CAPS lines or numbered sections
_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"#{1,4}\s.+?"           # Markdown headings: # H1, ## H2, ...
    r"|[A-Z][A-Z\s]{4,}$"   # ALL-CAPS lines (e.g. "INTRODUCTION", "RESULTS")
    r"|\d+\.\s+[A-Z][^\n]+"  # Numbered sections: "1. Introduction", "2.1 Methods"
    r")$",
    re.MULTILINE,
)


def _has_section_headers(text: str) -> bool:
    """Returns True if the text contains detectable section headings."""
    return bool(_HEADING_PATTERN.search(text))


def _split_by_sections(text: str) -> List[Dict[str, str]]:
    """
    Splits text into sections based on heading patterns.
    Returns list of {"title": str, "body": str} dicts.
    """
    lines = text.split("\n")
    sections: List[Dict[str, str]] = []
    current_title = "Introduction"
    current_lines: List[str] = []

    for line in lines:
        if _HEADING_PATTERN.match(line.strip()):
            # Save previous section
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({"title": current_title, "body": body})
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append({"title": current_title, "body": body})

    return sections if sections else [{"title": "", "body": text}]


def _token_split(text: str, chunk_size: int, chunk_overlap: int, enc) -> List[str]:
    """
    Simple token-budget recursive splitter used within each section.
    Splits on paragraph → sentence → word boundaries.
    """
    if _count_tokens(text, enc) <= chunk_size:
        return [text]

    # Try paragraph split first
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        chunks, current, current_tokens = [], [], 0
        for part in parts:
            part_with_sep = part + sep
            part_tokens = _count_tokens(part_with_sep, enc)
            if current_tokens + part_tokens > chunk_size and current:
                chunks.append(sep.join(current).strip())
                # Overlap: keep last few parts
                overlap, ov_tokens = [], 0
                for p in reversed(current):
                    pt = _count_tokens(p + sep, enc)
                    if ov_tokens + pt <= chunk_overlap:
                        overlap.insert(0, p)
                        ov_tokens += pt
                    else:
                        break
                current = overlap + [part]
                current_tokens = ov_tokens + part_tokens
            else:
                current.append(part)
                current_tokens += part_tokens
        if current:
            chunks.append(sep.join(current).strip())
        return [c for c in chunks if c.strip()]

    # Force-split by tokens as last resort
    tokens = enc.encode(text)
    step = max(1, chunk_size - chunk_overlap)
    return [
        enc.decode(tokens[i: i + chunk_size])
        for i in range(0, len(tokens), step)
    ]


def _section_aware_chunk(doc: Document, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """
    Splits text at section headings first, then applies token budget within each section.
    Each chunk carries the section_title in metadata.
    """
    enc = _get_encoding()
    sections = _split_by_sections(doc.text)
    chunks: List[Chunk] = []
    global_idx = 0

    for sec in sections:
        title = sec["title"]
        body = sec["body"]
        # Prepend heading to body so context is preserved in each chunk
        full_section = f"{title}\n\n{body}".strip() if title else body
        sub_texts = _token_split(full_section, chunk_size, chunk_overlap, enc)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue
            token_count = _count_tokens(sub_text, enc)
            metadata = {
                **doc.metadata,
                "chunk_index": global_idx,
                "section_title": title,
                "chunking_strategy": "section_aware",
            }
            chunk_id = _make_chunk_id(sub_text, metadata)
            chunks.append(Chunk(id=chunk_id, text=sub_text, metadata=metadata, token_count=token_count))
            global_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: Semantic Chunking (sentence-similarity based)
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """Splits text into individual sentences using regex."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _semantic_chunk(doc: Document, chunk_size: int, chunk_overlap: int,
                    similarity_threshold: float = 0.45) -> List[Chunk]:
    """
    Semantic chunking using sentence-transformers cosine similarity.
    Splits text where the similarity between consecutive sentences drops below threshold.
    Falls back to section-aware chunking if sentence-transformers unavailable.
    """
    try:
        from fastembed import TextEmbedding
        import numpy as np

        enc = _get_encoding()
        sentences = _split_into_sentences(doc.text)

        if len(sentences) <= 3:
            # Too short for semantic splitting — use section-aware
            return _section_aware_chunk(doc, chunk_size, chunk_overlap)

        # Load a lightweight model for similarity (already installed in project)
        _model_cache = _get_semantic_model()
        embeddings = list(_model_cache.embed(sentences))

        # Compute cosine similarity between consecutive sentences
        def cosine_sim(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        # Find split points: where similarity drops below threshold
        split_indices = [0]
        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < similarity_threshold:
                split_indices.append(i)
        split_indices.append(len(sentences))

        # Build semantic segments
        segments = []
        for i in range(len(split_indices) - 1):
            seg_sentences = sentences[split_indices[i]: split_indices[i + 1]]
            segments.append(" ".join(seg_sentences))

        # Now apply token budget within each segment
        chunks: List[Chunk] = []
        global_idx = 0
        for seg in segments:
            sub_texts = _token_split(seg, chunk_size, chunk_overlap, enc)
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                token_count = _count_tokens(sub_text, enc)
                metadata = {
                    **doc.metadata,
                    "chunk_index": global_idx,
                    "chunking_strategy": "semantic",
                }
                chunk_id = _make_chunk_id(sub_text, metadata)
                chunks.append(Chunk(id=chunk_id, text=sub_text, metadata=metadata, token_count=token_count))
                global_idx += 1
        return chunks

    except ImportError:
        logger.warning("fastembed not available. Falling back to section-aware chunking.")
        return _section_aware_chunk(doc, chunk_size, chunk_overlap)
    except Exception as e:
        logger.warning(f"Semantic chunking failed ({e}). Falling back to section-aware chunking.")
        return _section_aware_chunk(doc, chunk_size, chunk_overlap)


# Singleton semantic model cache
_semantic_model_instance = None

def _get_semantic_model():
    global _semantic_model_instance
    if _semantic_model_instance is None:
        from fastembed import TextEmbedding
        logger.info("Loading semantic chunking model (sentence-transformers/all-MiniLM-L6-v2)...")
        _semantic_model_instance = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Semantic chunking model loaded.")
    return _semantic_model_instance


# ---------------------------------------------------------------------------
# Legacy: Token Recursive Splitter (kept as ultimate fallback)
# ---------------------------------------------------------------------------

class TokenRecursiveCharacterSplitter:
    """Original token-budget recursive splitter — kept as ultimate fallback."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = _get_encoding(encoding_name)
        self.separators = ["\n\n", "\n", " ", ""]

    def _count_tokens(self, text: str) -> int:
        return _count_tokens(text, self.encoding)

    def _force_split(self, text: str, chunk_size: int) -> List[str]:
        tokens = self.encoding.encode(text)
        splits, step = [], max(1, chunk_size - self.chunk_overlap)
        for i in range(0, len(tokens), step):
            splits.append(self.encoding.decode(tokens[i: i + chunk_size]))
        return splits

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if self._count_tokens(text) <= self.chunk_size:
            return [text]
        if not separators:
            return self._force_split(text, self.chunk_size)
        separator = separators[0]
        splits = list(text) if separator == "" else text.split(separator)
        final_splits = [s + separator if i < len(splits) - 1 else s for i, s in enumerate(splits)]
        result = []
        for split in final_splits:
            if self._count_tokens(split) <= self.chunk_size:
                result.append(split)
            else:
                result.extend(self._split_text(split, separators[1:]))
        return result

    def split_text(self, text: str) -> List[str]:
        raw_splits = self._split_text(text, self.separators)
        chunks, current_parts, current_tokens = [], [], 0
        for split in raw_splits:
            split_tokens = self._count_tokens(split)
            if split_tokens > self.chunk_size:
                for fs in self._force_split(split, self.chunk_size):
                    fs_tokens = self._count_tokens(fs)
                    if current_tokens + fs_tokens > self.chunk_size and current_parts:
                        chunks.append("".join(current_parts))
                        ov, ov_t = self._get_overlap(current_parts)
                        current_parts, current_tokens = ov + [fs], ov_t + fs_tokens
                    else:
                        current_parts.append(fs)
                        current_tokens += fs_tokens
                continue
            if current_tokens + split_tokens > self.chunk_size and current_parts:
                chunks.append("".join(current_parts))
                ov, ov_t = self._get_overlap(current_parts)
                current_parts, current_tokens = ov + [split], ov_t + split_tokens
            else:
                current_parts.append(split)
                current_tokens += split_tokens
        if current_parts:
            chunks.append("".join(current_parts))
        return chunks

    def _get_overlap(self, parts: List[str]):
        overlap_parts, overlap_tokens = [], 0
        for part in reversed(parts):
            t = self._count_tokens(part)
            if overlap_tokens + t <= self.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_tokens += t
            else:
                break
        return overlap_parts, overlap_tokens


# ---------------------------------------------------------------------------
# Public API — Smart Chunking Router
# ---------------------------------------------------------------------------

def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Smart chunking router — selects the best strategy based on document content type.

    Routing logic:
      - content_type == 'table'  → Table-Atomic (never split tables)
      - content_type == 'image'  → Image-Atomic (single chunk from vision description)
      - Has section headings     → Section-Aware (split at headings, then token budget)
      - Otherwise                → Semantic (sentence-similarity based splitting)

    All strategies fall back gracefully and carry strategy name in chunk metadata.
    """
    content_type = doc.metadata.get("content_type", "text")

    # Table documents → atomic (never split)
    if content_type == "table":
        logger.debug(f"Chunking strategy: TABLE-ATOMIC for {doc.metadata.get('filename')}")
        return _chunk_table_document(doc)

    # Image documents → atomic (vision description as single chunk)
    if content_type == "image":
        logger.debug(f"Chunking strategy: IMAGE-ATOMIC for {doc.metadata.get('filename')}")
        return _chunk_image_document(doc)

    # Text documents: detect section headings
    if _has_section_headers(doc.text):
        logger.debug(f"Chunking strategy: SECTION-AWARE for {doc.metadata.get('filename')}")
        return _section_aware_chunk(doc, chunk_size, chunk_overlap)

    # Default: semantic chunking
    logger.debug(f"Chunking strategy: SEMANTIC for {doc.metadata.get('filename')}")
    return _semantic_chunk(doc, chunk_size, chunk_overlap)
