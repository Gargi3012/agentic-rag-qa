# Advanced Qdrant Search Guide

Qdrant is a self-hosted vector database designed for high-performance vector search. It allows storing payload metadata alongside vectors, making filtered vector queries highly efficient.

## Hybrid Search Config
To enable hybrid search in Qdrant, you must configure both dense and sparse vector indexes. Dense search models semantic meaning using cosine distance, while sparse search models keyword match frequencies like BM25.
In Qdrant, the prefetch API allows executing multiple search queries (one dense, one sparse) in parallel, and fusing their results using Reciprocal Rank Fusion (RRF).

## RRF Formula
RRF calculates a final relevance score by summing the reciprocal ranks of documents across both dense and sparse search results.
The fusion formula is:
Score = Sum( 1 / (rank + k) )
Where k is a constant parameter (often set to 60) that penalizes low-ranked documents.
This guarantees high-ranking consistency across diverse embedding spaces.
