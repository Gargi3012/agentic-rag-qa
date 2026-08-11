import os
import sys
import unittest

# Add workspace root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.chunker import TokenRecursiveCharacterSplitter, Chunk
from app.ingestion.dedup import generate_chunk_id, deduplicate_chunks
from app.eval.metrics import (
    calculate_recall,
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
    calculate_context_precision,
    calculate_exact_match,
    calculate_f1_score
)

class TestIngestionAndDeduplication(unittest.TestCase):
    def test_recursive_character_splitter(self):
        """
        Verify recursive token character splitting and token limits.
        """
        splitter = TokenRecursiveCharacterSplitter(chunk_size=15, chunk_overlap=3)
        text = "This is a simple sample text. It has multiple words and we split them recursively."
        
        # Test basic splitting
        splits = splitter.split_text(text)
        self.assertTrue(len(splits) > 1)
        
        # Verify that no chunk exceeds the 15 token limit
        for chunk in splits:
            tokens_count = len(splitter.encoding.encode(chunk))
            self.assertTrue(tokens_count <= 15)

    def test_generate_chunk_id_normalization(self):
        """
        Ensure Windows path drive casing and slash discrepancies resolve to identical UUIDs.
        """
        text = "Sample chunk of text for hashing."
        
        path_lower = r"d:\agentic_rag\data\file.md"
        path_upper = r"D:\agentic_rag\data\file.md"
        path_slashes = r"d:/agentic_rag/data/file.md"
        path_upper_slashes = r"D:/agentic_rag/data/file.md"
        
        id_lower = generate_chunk_id(text, path_lower)
        id_upper = generate_chunk_id(text, path_upper)
        id_slashes = generate_chunk_id(text, path_slashes)
        id_upper_slashes = generate_chunk_id(text, path_upper_slashes)
        
        # All of these variations must yield the exact same deterministic UUID
        self.assertEqual(id_lower, id_upper)
        self.assertEqual(id_lower, id_slashes)
        self.assertEqual(id_lower, id_upper_slashes)

    def test_deduplicate_chunks(self):
        """
        Verify that deduplicate_chunks filters out duplicate chunks in a batch.
        """
        metadata = {"filename": "test.md", "file_path": "d:/test.md"}
        chunk1 = Chunk(id="", text="Unique text A", metadata=metadata, token_count=3)
        chunk2 = Chunk(id="", text="Unique text B", metadata=metadata, token_count=3)
        chunk3 = Chunk(id="", text="Unique text A", metadata=metadata, token_count=3)  # Duplicate of chunk1
        
        batch = [chunk1, chunk2, chunk3]
        unique_batch = deduplicate_chunks(batch)
        
        self.assertEqual(len(unique_batch), 2)
        self.assertEqual(unique_batch[0].text, "Unique text A")
        self.assertEqual(unique_batch[1].text, "Unique text B")
        self.assertNotEqual(unique_batch[0].id, "")
        self.assertNotEqual(unique_batch[1].id, "")


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        # Setup standard lists for testing metrics
        # Relevant documents (Ground Truth)
        self.relevant_ids = ["doc1", "doc2"]
        # Retrieved documents in order
        self.retrieved_ids = ["doc3", "doc1", "doc4", "doc2", "doc5"]

    def test_recall_at_k(self):
        """
        Recall@k = (Hits in top-k) / (Total Relevant)
        """
        # Top 1 retrieved is ["doc3"] -> 0 hits
        self.assertEqual(calculate_recall(self.retrieved_ids, self.relevant_ids, k=1), 0.0)
        
        # Top 2 retrieved is ["doc3", "doc1"] -> 1 hit ("doc1") -> 1/2 = 0.5
        self.assertEqual(calculate_recall(self.retrieved_ids, self.relevant_ids, k=2), 0.5)
        
        # Top 4 retrieved is ["doc3", "doc1", "doc4", "doc2"] -> 2 hits -> 2/2 = 1.0
        self.assertEqual(calculate_recall(self.retrieved_ids, self.relevant_ids, k=4), 1.0)
        
        # Empty relevant case handling
        self.assertEqual(calculate_recall(self.retrieved_ids, [], k=2), 1.0)

    def test_hit_rate_at_k(self):
        """
        Hit Rate@k = 1 if hits in top-k > 0, else 0
        """
        self.assertEqual(calculate_hit_rate(self.retrieved_ids, self.relevant_ids, k=1), 0.0)
        self.assertEqual(calculate_hit_rate(self.retrieved_ids, self.relevant_ids, k=2), 1.0)
        self.assertEqual(calculate_hit_rate(self.retrieved_ids, [], k=2), 1.0)

    def test_mean_reciprocal_rank(self):
        """
        MRR = 1 / rank of the first relevant document retrieved
        First relevant doc is "doc1" at rank 2 (index 1) -> MRR = 1/2 = 0.5
        """
        self.assertEqual(calculate_mrr(self.retrieved_ids, self.relevant_ids), 0.5)
        
        # No hits case -> MRR = 0.0
        self.assertEqual(calculate_mrr(["doc3", "doc4"], self.relevant_ids), 0.0)

    def test_ndcg_at_k(self):
        """
        nDCG@k = DCG@k / IDCG@k
        For k=3: retrieved = ["doc3", "doc1", "doc4"] -> relevance = [0, 1, 0]
        DCG@3 = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.6309
        Ideal sorted = ["doc1", "doc2", "doc3"] -> relevance = [1, 1, 0]
        IDCG@3 = 1/log2(2) + 1/log2(3) = 1.6309
        nDCG@3 = 0.6309 / 1.6309 = 0.3868
        """
        ndcg_3 = calculate_ndcg(self.retrieved_ids, self.relevant_ids, k=3)
        self.assertAlmostEqual(ndcg_3, 0.38685, places=4)
        
        # Perfect order case -> nDCG = 1.0
        self.assertEqual(calculate_ndcg(["doc1", "doc2"], self.relevant_ids, k=2), 1.0)

    def test_context_precision_at_k(self):
        """
        Context Precision@k = Sum(Precision@i * rel_i) / (hits in k)
        For k=4: retrieved = ["doc3", "doc1", "doc4", "doc2"]
        Rank 1: "doc3" (irrelevant) -> Precision@1 = 0
        Rank 2: "doc1" (relevant)   -> Precision@2 = 1/2 = 0.5
        Rank 3: "doc4" (irrelevant) -> Precision@3 = 1/3 = 0.333
        Rank 4: "doc2" (relevant)   -> Precision@4 = 2/4 = 0.5
        Sum = (0.5 * 1) + (0.5 * 1) = 1.0
        Total hits in 4 = 2
        Context Precision@4 = 1.0 / 2 = 0.5
        """
        precision_4 = calculate_context_precision(self.retrieved_ids, self.relevant_ids, k=4)
        self.assertEqual(precision_4, 0.5)

    def test_traditional_text_matches(self):
        """
        Test Exact Match (EM) and word-level F1-score comparisons.
        """
        pred = "FastAPI is built on Starlette and Pydantic."
        gold = "FastAPI is built on Starlette and Pydantic!"
        
        # EM normalizes and ignores punctuation
        self.assertEqual(calculate_exact_match(pred, gold), 1.0)
        self.assertEqual(calculate_exact_match("Different text", gold), 0.0)
        
        # F1 score calculations
        f1 = calculate_f1_score("FastAPI is built on Starlette", "FastAPI is built on Starlette and Pydantic")
        # 5 common words out of 5 and 7 words
        # Precision = 5/5 = 1.0
        # Recall = 5/7 = 0.714
        # F1 = 2 * (1.0 * 0.714) / (1.0 + 0.714) = 1.428 / 1.714 = 0.833
        self.assertAlmostEqual(f1, 0.83333, places=4)

if __name__ == "__main__":
    unittest.main()
