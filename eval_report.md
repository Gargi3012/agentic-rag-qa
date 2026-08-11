# Agentic RAG Evaluation Report

This report summarizes the performance, retrieval quality, response quality, latency, and costs computed for the **Cost-Efficient Agentic RAG QA Service** over a fixed dataset of `18` test queries.

---

## 📊 Summary Metrics Table

| Metric Category | Metric Name | Score / Value | Description |
| :--- | :--- | :---: | :--- |
| **Retrieval** | Recall@5 | `0.7778` | Percent of relevant context chunks retrieved in top-5 |
| | Mean Reciprocal Rank (MRR) | `0.7778` | Rank quality of the first relevant chunk |
| | nDCG@5 | `0.7778` | Normalized Discounted Cumulative Gain ranking quality |
| | Context Precision@5 | `0.7778` | Score of relevant chunks ordered correctly at top-5 |
| **Generation** | Exact Match (EM) | `16.6667%` | Strict text match against reference gold answers |
| | F1 Score | `0.5614` | Word-level token overlap score |
| **LLM-as-a-Judge** | Faithfulness | `5.00 / 5.00` | Groundedness of response based ONLY on context |
| | Answer Relevance | `5.00 / 5.00` | How well the generated response answers the query |
| **Telemetry** | Total Cost (USD) | `$0.003682` | Combined cost for OpenAI calls during run |
| | Avg Cost per Query | `$0.000205` | Average expense per execution |

---

## ⚡ Latency Performance (ms)

| Pipeline Phase | p50 (Median) | p95 (95th Percentile) |
| :--- | :---: | :---: |
| **Retrieval Only** (Analysis + Hybrid Search) | `1521.87 ms` | `2502.32 ms` |
| **Full Pipeline** (Retrieval + Rerank + Gen + Critic) | `4408.82 ms` | `7805.82 ms` |

---

## 💳 Vector DB Cost Comparison (Self-Hosted vs Managed)

The following table compares the monthly infrastructure cost of running a self-hosted Qdrant instance (on AWS EC2) versus standard managed vector database tiers (Pinecone Serverless / Qdrant Cloud) at different vector scale thresholds.

### Cost Table (USD/Month)

| Vector Volume | Self-Hosted Qdrant (Docker on AWS) | Managed Vector DB (Qdrant Cloud / Pinecone) | Cost Savings (Self-Hosted) |
| :---: | :--- | :--- | :---: |
| **100K Chunks** | **$15.00 / month** <br> *AWS t3.small (2GB RAM)* | **$0.00 - $10.00 / month** <br> *Free tier / Minimum Serverless* | Managed is cheaper at small scale |
| **1M Chunks** | **$30.00 / month** <br> *AWS t3.medium (4GB RAM)* | **$45.00 / month** <br> *Qdrant Cloud Starter Plan* | **~33% Savings** |
| **10M Chunks** | **$110.00 / month** <br> *AWS r6g.large (16GB RAM + EBS)* | **$180.00 / month** <br> *Qdrant Cloud Standard Tier* | **~39% Savings** |

### Pricing Assumptions & Architectural Trade-offs
1. **Infrastructure Specs**:
   - Chunks average **2.5KB** payload (512 tokens + metadata path/author tags).
   - Dense vectors: 384 dimensions (`all-MiniLM-L6-v2`), requiring 1.5KB raw floats storage.
   - Sparse vectors: BM25 representation (highly compressed indices).
2. **Self-Hosted Pricing Rationale**:
   - Powered by AWS EC2 standard instances using Docker Compose volumes.
   - Storage fits in memory for rapid index checks; EBS gp3 storage handles writes.
3. **Managed Cloud Tiers**:
   - Assumes Qdrant Cloud standard instance configurations with redundant replicas.
   - Pinecone serverless calculated based on write units ($1/million) and read queries.
4. **Scale Inflection Point**:
   - At **<100K vectors**, managed/free tiers are highly economical.
   - At **>1M vectors**, self-hosting Qdrant saves substantial operational markup, making it the most cost-efficient choice for production enterprise pipelines.
