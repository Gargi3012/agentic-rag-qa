# Advanced Retrieval-Augmented Generation (RAG) Techniques

## 1. Parent-Child Chunking
Parent-Child Chunking is an advanced RAG technique. In this approach, we split documents into larger parent chunks (e.g., 1024 tokens) to retain broad semantic context, and smaller nested child chunks (e.g., 128 tokens) for granular vector search. When a child chunk matches the user query, we retrieve and pass the larger parent chunk to the LLM. This provides a balance between high-fidelity retrieval and complete context.

## 2. Hierarchical Indexing
Hierarchical Indexing structures documents into multi-level nodes. Larger parent nodes represent summaries or chapters, while child nodes represent specific paragraphs. Similar to Parent-Child chunking, retrieval matches against leaf child nodes, but is expanded to parent nodes during context loading. This helps prevent context loss in complex documents.
