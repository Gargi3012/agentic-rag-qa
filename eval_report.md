# Agentic RAG Evaluation Report

This report summarizes the performance, retrieval quality, response quality, latency, and costs computed for the **Cost-Efficient Agentic RAG QA Service** over a fixed dataset of `20` test queries.

---

## 📊 Summary Metrics Table

| Metric Category | Metric Name | Score / Value | Description |
| :--- | :--- | :---: | :--- |
| **On-Domain Retrieval** | Recall@5 | `0.9412` | Percent of relevant context chunks retrieved in top-5 (On-Domain only) |
| | Mean Reciprocal Rank (MRR) | `0.9118` | Rank quality of the first relevant chunk (On-Domain only) |
| | nDCG@5 | `0.9255` | Normalized Discounted Cumulative Gain ranking quality (On-Domain only) |
| | Context Precision@5 | `0.9199` | Score of relevant chunks ordered correctly at top-5 (On-Domain only) |
| **Guardrails** | Relevance Gate Accuracy | `100.00%` | Correct refusal rate for out-of-domain queries (threshold = 0.35) |
| **Generation** | Exact Match (EM) | `15.0000%` | Strict text match against reference gold answers (On-Domain only) |
| | F1 Score | `0.5426` | Word-level token overlap score |
| **LLM-as-a-Judge** | Faithfulness | `4.95 / 5.00` | Groundedness of response based ONLY on context |
| | Answer Relevance | `5.00 / 5.00` | How well the generated response answers the query |
| **Telemetry** | Total Cost (USD) | `$0.005836` | Combined cost for OpenAI calls during run |
| | Avg Cost per Query | `$0.000292` | Average expense per execution |

> [!NOTE]
> **On-Domain Metrics Explanation**: Previously, on-domain retrieval metrics (Recall@5, MRR, nDCG@5, Context Precision@5) showed identical values due to a tiny 3-document corpus with binary (found/not found at Rank 1) outcomes. By expanding the corpus (9+ documents, including hard negatives) and introducing multi-target queries, we introduced realistic ranking variations that produce mathematically distinct, authentic metrics.

---

## 🚪 Out-of-Domain Guardrail Details

| Query ID | Out-of-Domain Query | Best Rerank Score | Gate Threshold | Status |
| :---: | :--- | :---: | :---: | :---: |
| q16 | 'Who is the president of France?' | `0.0000` | `0.35` | ✅ Correctly Rejected |
| q17 | 'What is the capital of Japan?' | `0.0000` | `0.35` | ✅ Correctly Rejected |
| q18 | 'How many moons does Mars have?' | `0.0003` | `0.35` | ✅ Correctly Rejected |

---

## ⚡ Latency Performance (ms)

| Pipeline Phase | p50 (Median) | p95 (95th Percentile) |
| :--- | :---: | :---: |
| **Retrieval Only** (Analysis + Hybrid Search) | `2658.35 ms` | `3714.98 ms` |
| **Full Pipeline** (Retrieval + Rerank + Gen + Critic) | `5221.81 ms` | `7896.46 ms` |

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
   - gp3 storage handles writes.
3. **Managed Cloud Tiers**:
   - Assumes Qdrant Cloud standard instance configurations with redundant replicas.
   - Pinecone serverless calculated based on write units ($1/million) and read queries.
4. **Scale Inflection Point**:
   - At **<100K vectors**, managed/free tiers are highly economical.
   - At **>1M vectors**, self-hosting Qdrant saves substantial operational markup, making it the most cost-efficient choice for production enterprise pipelines.
