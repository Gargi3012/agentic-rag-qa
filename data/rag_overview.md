# RAG Architecture and Critique Loops

Retrieval-Augmented Generation (RAG) combines search retrieval with generative LLMs to answer questions using external domain context.

## Chunking Strategy
Recursive chunking splits large text documents by semantic separators (like double newlines, single newlines, or spaces) to keep individual parts within a token budget.
A target chunk size of 512 tokens with a 50 token overlap ensures that context transitions are preserved between neighboring chunks.

## Cross-Encoder Reranking
Standard dense vector search is fast but lacks detailed keyword-document cross-attention. Integrating a Cross-Encoder (like `ms-marco-MiniLM-L-6-v2`) re-ranks the top retrieval hits.
The Cross-Encoder processes the query and document text together, yielding log-odds scores that can be normalized to a [0.0, 1.0] range via a sigmoid function.

## Critic Pass & Retries
To eliminate hallucinations, a RAG system can employ a Critique Pass. A secondary cheap LLM pass evaluates if the generated answer is grounded in the retrieved text.
If the critic detects hallucinated facts, it triggers a strict regeneration retry. The pipeline enforces a hard cap of 1 retry to avoid infinite execution loops.
If the retry still fails validation, the answer is returned with a low confidence flag.
