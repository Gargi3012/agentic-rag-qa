import logging
import tiktoken
from typing import List, Dict, Any
from pydantic import BaseModel
from app.ingestion.loader import Document

logger = logging.getLogger("agentic_rag.ingestion.chunker")

class Chunk(BaseModel):
    id: str  # Generated via hash
    text: str
    metadata: Dict[str, Any]
    token_count: int

class TokenRecursiveCharacterSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(f"Could not load encoding {encoding_name}, falling back to cl100k_base.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.separators = ["\n\n", "\n", " ", ""]

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _force_split(self, text: str, chunk_size: int) -> List[str]:
        """
        Split a string by token offsets when it cannot be split by any separator.
        """
        tokens = self.encoding.encode(text)
        splits = []
        step = max(1, chunk_size - self.chunk_overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            splits.append(self.encoding.decode(chunk_tokens))
        return splits

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using a list of separators.
        """
        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._force_split(text, self.chunk_size)

        separator = separators[0]
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Recombine splits with their separator
        final_splits = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                final_splits.append(split + separator)
            else:
                final_splits.append(split)

        result = []
        for split in final_splits:
            if self._count_tokens(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recurse with the next separator
                result.extend(self._split_text(split, separators[1:]))

        return result

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text and merges adjacent splits to optimize chunk sizes.
        """
        # Step 1: Split the text recursively into smaller semantic units
        raw_splits = self._split_text(text, self.separators)
        
        # Step 2: Merge splits while keeping them within the token budget
        chunks = []
        current_chunk_parts = []
        current_chunk_tokens = 0

        for split in raw_splits:
            split_tokens = self._count_tokens(split)
            
            # If a single split exceeds chunk_size, force-split it first
            if split_tokens > self.chunk_size:
                force_splits = self._force_split(split, self.chunk_size)
                for fs in force_splits:
                    # Treat each force split as an independent item
                    fs_tokens = self._count_tokens(fs)
                    if current_chunk_tokens + fs_tokens > self.chunk_size:
                        if current_chunk_parts:
                            chunks.append("".join(current_chunk_parts))
                        # Compute overlap
                        overlap_parts, overlap_tokens = self._get_overlap_splits(current_chunk_parts)
                        current_chunk_parts = overlap_parts + [fs]
                        current_chunk_tokens = overlap_tokens + fs_tokens
                    else:
                        current_chunk_parts.append(fs)
                        current_chunk_tokens += fs_tokens
                continue

            if current_chunk_tokens + split_tokens > self.chunk_size:
                if current_chunk_parts:
                    chunks.append("".join(current_chunk_parts))
                
                # Retrieve the overlap suffix
                overlap_parts, overlap_tokens = self._get_overlap_splits(current_chunk_parts)
                current_chunk_parts = overlap_parts + [split]
                current_chunk_tokens = overlap_tokens + split_tokens
            else:
                current_chunk_parts.append(split)
                current_chunk_tokens += split_tokens

        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts))

        return chunks

    def _get_overlap_splits(self, parts: List[str]) -> tuple[List[str], int]:
        """
        Helper to construct overlapping suffix from the current chunk.
        """
        overlap_parts = []
        overlap_tokens = 0
        for part in reversed(parts):
            tokens = self._count_tokens(part)
            if overlap_tokens + tokens <= self.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_tokens += tokens
            else:
                break
        return overlap_parts, overlap_tokens

def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 50) -> List[Chunk]:
    """
    Chunks a Document object and assigns metadata to each chunk, counting its tokens.
    """
    splitter = TokenRecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_text(doc.text)
    
    chunks = []
    for idx, text in enumerate(text_chunks):
        chunk_metadata = doc.metadata.copy()
        chunk_metadata["chunk_index"] = idx
        
        # We don't generate the SHA-256 ID here to keep chunking logic independent of hash-dedup implementation
        token_count = splitter._count_tokens(text)
        
        # Create chunk
        chunks.append(Chunk(
            id="", # Will be set by deduplicator
            text=text,
            metadata=chunk_metadata,
            token_count=token_count
        ))
    return chunks
