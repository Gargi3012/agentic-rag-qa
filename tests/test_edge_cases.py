"""
Comprehensive edge-case test suite for the Advanced RAG Pipeline.
Tests: loader.py, chunker.py, image_understander.py, agent.py
Run: python -m pytest tests/test_edge_cases.py -v
"""

import os
import sys
import tempfile
import textwrap
import pytest

# Make sure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =============================================================================
# SECTION 1: loader.py edge cases
# =============================================================================

class TestTableToMarkdown:
    """Tests for the _table_to_markdown helper."""

    def _get_fn(self):
        from app.ingestion.loader import _table_to_markdown
        return _table_to_markdown

    def test_normal_table(self):
        fn = self._get_fn()
        table = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = fn(table)
        assert "| Name | Age |" in md
        assert "| Alice | 30 |" in md
        assert "---" in md

    def test_none_cells_replaced_with_empty(self):
        fn = self._get_fn()
        table = [["Col1", "Col2"], [None, "val2"], ["val3", None]]
        md = fn(table)
        assert "Col1" in md
        assert "val2" in md

    def test_empty_table_returns_empty_string(self):
        fn = self._get_fn()
        assert fn([]) == ""
        assert fn([[]]) == ""

    def test_single_row_table(self):
        """Table with only a header and no data rows."""
        fn = self._get_fn()
        table = [["Header1", "Header2"]]
        md = fn(table)
        assert "| Header1 | Header2 |" in md
        assert "---" in md

    def test_row_shorter_than_header(self):
        """Rows shorter than header should be padded with empty cells."""
        fn = self._get_fn()
        table = [["A", "B", "C"], ["x", "y"]]  # missing last cell
        md = fn(table)
        lines = md.strip().split("\n")
        # All rows should have same number of pipes
        pipe_counts = [line.count("|") for line in lines]
        assert len(set(pipe_counts)) == 1, "All rows should have same column count"

    def test_single_column_table(self):
        fn = self._get_fn()
        table = [["Item"], ["Apple"], ["Banana"]]
        md = fn(table)
        assert "| Item |" in md
        assert "| Apple |" in md

    def test_special_characters_in_cells(self):
        fn = self._get_fn()
        table = [["Formula", "Result"], ["a+b=c", "$1,000"]]
        md = fn(table)
        assert "a+b=c" in md
        assert "$1,000" in md


class TestLoadFile:
    """Tests for load_file() with various file types and edge cases."""

    def _get_fn(self):
        from app.ingestion.loader import load_file
        return load_file

    def test_nonexistent_file_returns_empty_list(self):
        load_file = self._get_fn()
        result = load_file("/path/that/does/not/exist.pdf")
        assert result == []

    def test_markdown_file_returns_single_document(self):
        load_file = self._get_fn()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write("# Hello\n\nThis is a test markdown file.\n")
            path = f.name
        try:
            docs = load_file(path)
            assert len(docs) == 1
            assert "Hello" in docs[0].text
            assert docs[0].metadata["content_type"] == "text"
            assert docs[0].metadata["file_type"] == "markdown"
        finally:
            os.unlink(path)

    def test_html_file_strips_scripts_and_styles(self):
        load_file = self._get_fn()
        html = """<html><head><script>alert('x')</script></head>
        <body><h1>Title</h1><p>Content here</p></body></html>"""
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                          delete=False, encoding="utf-8") as f:
            f.write(html)
            path = f.name
        try:
            docs = load_file(path)
            assert len(docs) == 1
            assert "alert" not in docs[0].text
            assert "Title" in docs[0].text
            assert "Content here" in docs[0].text
        finally:
            os.unlink(path)

    def test_empty_markdown_file_returns_empty_list(self):
        load_file = self._get_fn()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                          delete=False, encoding="utf-8") as f:
            f.write("   \n\n  \n")  # only whitespace
            path = f.name
        try:
            docs = load_file(path)
            assert docs == []
        finally:
            os.unlink(path)

    def test_txt_file_as_plaintext_fallback(self):
        load_file = self._get_fn()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                          delete=False, encoding="utf-8") as f:
            f.write("Plain text document content.")
            path = f.name
        try:
            docs = load_file(path)
            assert len(docs) == 1
            assert "Plain text" in docs[0].text
            assert docs[0].metadata["file_type"] == "text"
        finally:
            os.unlink(path)

    def test_metadata_fields_present(self):
        load_file = self._get_fn()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                          delete=False, encoding="utf-8") as f:
            f.write("# Test\nSome content.")
            path = f.name
        try:
            docs = load_file(path)
            meta = docs[0].metadata
            assert "filename" in meta
            assert "file_path" in meta
            assert "file_size" in meta
            assert "last_modified" in meta
            assert "content_type" in meta
            assert "file_type" in meta
        finally:
            os.unlink(path)

    def test_returns_list_not_optional(self):
        """Ensure load_file always returns a list, never None."""
        load_file = self._get_fn()
        result = load_file("/nonexistent.xyz")
        assert isinstance(result, list)


class TestLoadDirectory:
    """Tests for load_directory()."""

    def _get_fn(self):
        from app.ingestion.loader import load_directory
        return load_directory

    def test_nonexistent_directory_returns_empty(self):
        load_directory = self._get_fn()
        assert load_directory("/nonexistent/path/xyz") == []

    def test_empty_directory_returns_empty(self):
        load_directory = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            result = load_directory(tmp)
            assert result == []

    def test_mixed_files_loaded(self):
        load_directory = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            # Create 2 supported files
            with open(os.path.join(tmp, "a.md"), "w", encoding="utf-8") as f:
                f.write("# Doc A\nContent A")
            with open(os.path.join(tmp, "b.txt"), "w", encoding="utf-8") as f:
                f.write("Content B")
            # Create 1 unsupported binary file (should not crash)
            with open(os.path.join(tmp, "img.png"), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            docs = load_directory(tmp)
            assert len(docs) >= 2
            filenames = [d.metadata["filename"] for d in docs]
            assert "a.md" in filenames
            assert "b.txt" in filenames

    def test_recursive_subdirectory_scan(self):
        load_directory = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "sub")
            os.makedirs(sub)
            with open(os.path.join(sub, "nested.md"), "w", encoding="utf-8") as f:
                f.write("# Nested\nNested content.")
            docs = load_directory(tmp)
            assert any(d.metadata["filename"] == "nested.md" for d in docs)


# =============================================================================
# SECTION 2: chunker.py edge cases
# =============================================================================

class TestTableAwareChunking:
    """Table documents must be atomic (never split)."""

    def _get_fn(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        return chunk_document, Document

    def test_table_document_produces_exactly_one_chunk(self):
        chunk_document, Document = self._get_fn()
        big_table = "\n".join(
            f"| Col1 | Col2 | Col3 |\n| val{i} | val{i+1} | val{i+2} |"
            for i in range(50)
        )
        doc = Document(text=f"[TABLE -- Page 1]\n\n{big_table}",
                       metadata={"filename": "test.pdf", "content_type": "table",
                                 "file_path": "/test.pdf"})
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)  # tiny chunk_size
        assert len(chunks) == 1, "Table must never be split"
        assert chunks[0].metadata["chunking_strategy"] == "table_atomic"

    def test_table_chunk_has_correct_metadata(self):
        chunk_document, Document = self._get_fn()
        doc = Document(text="[TABLE -- Page 2]\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
                       metadata={"filename": "report.pdf", "content_type": "table",
                                 "file_path": "/report.pdf", "table_page": 2})
        chunks = chunk_document(doc)
        assert chunks[0].metadata["content_type"] == "table"
        assert chunks[0].id != ""  # ID must be set


class TestImageAwareChunking:
    """Image documents must be atomic (single chunk per image)."""

    def test_image_document_produces_exactly_one_chunk(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(
            text="[FIGURE -- report.pdf, Page 3, Figure 0]\n\nDescription: A bar chart showing quarterly revenue.\n\nLabels: Q1, Q2, Q3\n\nTrend: Revenue grew 30%.",
            metadata={"filename": "report.pdf", "content_type": "image",
                      "file_path": "/report.pdf", "image_page": 3}
        )
        chunks = chunk_document(doc, chunk_size=20)  # tiny chunk_size
        assert len(chunks) == 1
        assert chunks[0].metadata["chunking_strategy"] == "image_atomic"


class TestSectionAwareChunking:
    """Documents with headings should be split at section boundaries."""

    def test_markdown_headings_detected(self):
        from app.ingestion.chunker import _has_section_headers
        text = "# Introduction\n\nSome content here.\n\n## Methods\n\nMore content."
        assert _has_section_headers(text) is True

    def test_no_headings_not_detected(self):
        from app.ingestion.chunker import _has_section_headers
        text = "Just regular prose text with no headings whatsoever."
        assert _has_section_headers(text) is False

    def test_allcaps_section_detected(self):
        from app.ingestion.chunker import _has_section_headers
        text = "INTRODUCTION\n\nSome content.\n\nRESULTS\n\nMore content."
        assert _has_section_headers(text) is True

    def test_section_title_in_chunk_metadata(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        text = "# Introduction\n\nThis is the intro. " * 5
        doc = Document(text=text,
                       metadata={"filename": "paper.pdf", "content_type": "text",
                                 "file_path": "/paper.pdf"})
        chunks = chunk_document(doc)
        strategies = {c.metadata.get("chunking_strategy") for c in chunks}
        assert "section_aware" in strategies
        for c in chunks:
            assert "section_title" in c.metadata

    def test_multiple_sections_split_correctly(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        # Create a doc with 3 distinct sections
        text = (
            "# Introduction\n\n" + "This introduces the topic. " * 20 + "\n\n"
            "# Methods\n\n" + "Here are the methods used. " * 20 + "\n\n"
            "# Results\n\n" + "The results show that. " * 20
        )
        doc = Document(text=text,
                       metadata={"filename": "paper.pdf", "content_type": "text",
                                 "file_path": "/paper.pdf"})
        chunks = chunk_document(doc)
        assert len(chunks) >= 3  # At minimum one chunk per section


class TestSemanticChunking:
    """Semantic chunking for plain text without headings."""

    def test_short_text_returns_at_least_one_chunk(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(text="This is a short document.",
                       metadata={"filename": "short.txt", "content_type": "text",
                                 "file_path": "/short.txt"})
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].text.strip() != ""

    def test_empty_text_returns_empty_list(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(text="   ",
                       metadata={"filename": "empty.txt", "content_type": "text",
                                 "file_path": "/empty.txt"})
        chunks = chunk_document(doc)
        assert chunks == []

    def test_long_text_produces_multiple_chunks(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        # 2000 words of text that should split into multiple chunks
        long_text = ("The quick brown fox jumps over the lazy dog. " * 200)
        doc = Document(text=long_text,
                       metadata={"filename": "long.txt", "content_type": "text",
                                 "file_path": "/long.txt"})
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_all_chunks_have_required_fields(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(text="Hello world. This is test content. " * 20,
                       metadata={"filename": "test.md", "content_type": "text",
                                 "file_path": "/test.md"})
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.id != "", f"Chunk ID must not be empty: {c}"
            assert c.text.strip() != "", "Chunk text must not be empty"
            assert c.token_count > 0, "Token count must be positive"
            assert "filename" in c.metadata
            assert "chunk_index" in c.metadata
            assert "chunking_strategy" in c.metadata

    def test_chunk_ids_are_unique(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(text="Sentence one. Sentence two. Sentence three. " * 50,
                       metadata={"filename": "dup_test.md", "content_type": "text",
                                 "file_path": "/dup_test.md"})
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=5)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"

    def test_chunk_indices_sequential(self):
        from app.ingestion.chunker import chunk_document
        from app.ingestion.loader import Document
        doc = Document(text="Content. " * 100,
                       metadata={"filename": "seq.md", "content_type": "text",
                                 "file_path": "/seq.md"})
        chunks = chunk_document(doc, chunk_size=30, chunk_overlap=5)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices must be sequential"


# =============================================================================
# SECTION 3: image_understander.py edge cases
# =============================================================================

class TestBuildImageDocumentText:

    def test_full_vision_result(self):
        from app.ingestion.image_understander import build_image_document_text
        meta = {"image_page": 5, "image_index": 2, "filename": "annual_report.pdf"}
        vision = {
            "description": "A pie chart showing market share distribution.",
            "labels": ["CompanyA", "CompanyB", "CompanyC"],
            "trend_summary": "CompanyA dominates with 60% market share."
        }
        text = build_image_document_text(meta, vision)
        assert "Page 5" in text
        assert "CompanyA" in text
        assert "dominates" in text
        assert "pie chart" in text

    def test_empty_labels_handled(self):
        from app.ingestion.image_understander import build_image_document_text
        meta = {"image_page": 1, "image_index": 0, "filename": "doc.pdf"}
        vision = {"description": "A photograph.", "labels": [], "trend_summary": "N/A"}
        text = build_image_document_text(meta, vision)
        assert "none detected" in text
        assert "N/A" in text

    def test_missing_keys_dont_crash(self):
        from app.ingestion.image_understander import build_image_document_text
        meta = {}  # empty metadata
        vision = {}  # empty vision result
        text = build_image_document_text(meta, vision)
        assert isinstance(text, str)

    def test_no_vision_clients_returns_fallback(self):
        from app.ingestion.image_understander import describe_image
        import base64
        # 1x1 white PNG
        tiny_png = base64.b64encode(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
            b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
            b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        ).decode("utf-8")
        result = describe_image(tiny_png, "png", openai_client=None, groq_client=None)
        assert "description" in result
        assert "labels" in result
        assert "trend_summary" in result
        # Should return fallback dict, not crash
        assert isinstance(result["labels"], list)

    def test_get_vision_clients_no_keys_returns_none(self):
        """When no API keys set, both clients should be None."""
        import os
        from app.ingestion.image_understander import get_vision_clients
        # Temporarily clear keys
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_groq = os.environ.pop("GROQ_API_KEY", None)
        try:
            openai_c, groq_c = get_vision_clients()
            assert openai_c is None
            assert groq_c is None
        finally:
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai
            if old_groq:
                os.environ["GROQ_API_KEY"] = old_groq


# =============================================================================
# SECTION 4: agent.py confidence score edge cases
# =============================================================================

class TestComputeConfidenceScore:

    def _get_fn(self):
        from app.generation.agent import compute_confidence_score
        return compute_confidence_score

    def test_perfect_score(self):
        fn = self._get_fn()
        score = fn(rerank_score=1.0, critic_grounded=True, retries=0)
        assert score == 1.0

    def test_zero_rerank_grounded_no_retry(self):
        fn = self._get_fn()
        score = fn(rerank_score=0.0, critic_grounded=True, retries=0)
        assert score == pytest.approx(0.60, abs=0.01)

    def test_retry_reduces_score(self):
        fn = self._get_fn()
        score_no_retry = fn(rerank_score=0.8, critic_grounded=True, retries=0)
        score_with_retry = fn(rerank_score=0.8, critic_grounded=True, retries=1)
        assert score_no_retry > score_with_retry

    def test_ungrounded_reduces_score(self):
        fn = self._get_fn()
        score_grounded = fn(rerank_score=0.8, critic_grounded=True, retries=0)
        score_ungrounded = fn(rerank_score=0.8, critic_grounded=False, retries=0)
        assert score_grounded > score_ungrounded

    def test_worst_case_score(self):
        fn = self._get_fn()
        score = fn(rerank_score=0.0, critic_grounded=False, retries=1)
        assert score == 0.0

    def test_score_always_in_range(self):
        fn = self._get_fn()
        test_cases = [
            (0.0, False, 0),
            (0.0, False, 1),
            (1.0, True, 0),
            (1.0, True, 1),
            (0.5, True, 1),
            (0.5, False, 0),
            (0.3, True, 0),
            (0.9, False, 1),
        ]
        for rerank, grounded, retries in test_cases:
            score = fn(rerank, grounded, retries)
            assert 0.0 <= score <= 1.0, f"Score out of range for ({rerank},{grounded},{retries}): {score}"

    def test_score_is_float_with_4_decimal_places(self):
        fn = self._get_fn()
        score = fn(rerank_score=0.7654321, critic_grounded=True, retries=0)
        assert isinstance(score, float)
        # Should be rounded to 4 decimal places
        assert score == round(score, 4)

    def test_high_rerank_low_confidence_grounding(self):
        """Good retrieval but critic failed — medium score."""
        fn = self._get_fn()
        score = fn(rerank_score=1.0, critic_grounded=False, retries=1)
        # 0.40*1.0 + 0.40*0.0 + 0.20*0.0 = 0.40
        assert score == pytest.approx(0.40, abs=0.01)


# =============================================================================
# SECTION 5: dedup.py edge cases
# =============================================================================

class TestDeduplication:

    def test_same_chunk_deduplicated(self):
        from app.ingestion.dedup import deduplicate_chunks
        from app.ingestion.chunker import Chunk
        chunk = Chunk(id="", text="Duplicate text", metadata={"filename": "a.pdf", "file_path": "/a.pdf"}, token_count=3)
        chunks = [chunk, chunk]
        unique = deduplicate_chunks(chunks)
        assert len(unique) == 1

    def test_different_chunks_kept(self):
        from app.ingestion.dedup import deduplicate_chunks
        from app.ingestion.chunker import Chunk
        c1 = Chunk(id="", text="First unique chunk", metadata={"filename": "a.pdf", "file_path": "/a.pdf"}, token_count=3)
        c2 = Chunk(id="", text="Second unique chunk", metadata={"filename": "a.pdf", "file_path": "/a.pdf"}, token_count=3)
        unique = deduplicate_chunks([c1, c2])
        assert len(unique) == 2

    def test_empty_list_handled(self):
        from app.ingestion.dedup import deduplicate_chunks
        assert deduplicate_chunks([]) == []

    def test_ids_are_deterministic(self):
        """Re-running dedup on same input should produce identical IDs."""
        from app.ingestion.dedup import deduplicate_chunks
        from app.ingestion.chunker import Chunk
        c = Chunk(id="", text="Deterministic text", metadata={"filename": "test.pdf", "file_path": "/test.pdf"}, token_count=3)
        first_run = deduplicate_chunks([c])
        second_run = deduplicate_chunks([c])
        assert first_run[0].id == second_run[0].id

    def test_generate_chunk_id_different_paths(self):
        """Same text but different file paths should produce different IDs."""
        from app.ingestion.dedup import generate_chunk_id
        id1 = generate_chunk_id("same text", "/path/file_a.pdf")
        id2 = generate_chunk_id("same text", "/path/file_b.pdf")
        assert id1 != id2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
