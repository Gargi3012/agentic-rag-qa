import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.config import Config
from app.ingestion import load_directory, chunk_document, deduplicate_chunks
from app.retrieval.qdrant_client import QdrantStore
from app.generation.agent import AgenticQueryPipeline
from app.eval.metrics import (
    calculate_recall,
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
    calculate_context_precision,
    calculate_exact_match,
    calculate_f1_score,
    run_llm_judge_eval
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag.eval.harness")

def ensure_corpus_ingested(store: QdrantStore):
    """
    Forces recreation of the evaluation collection and ingests the evaluation corpus.
    """
    collection_name = "agentic_rag_eval"
    store.collection_name = collection_name
    
    # Force recreate collection to avoid data contamination
    try:
        if store.client.collection_exists(collection_name=collection_name):
            logger.info(f"Deleting existing evaluation collection '{collection_name}'...")
            store.client.delete_collection(collection_name=collection_name)
    except Exception as e:
        logger.warning(f"Failed to check/delete collection: {str(e)}")

    store.init_collection(collection_name)
    
    logger.info("Ingesting fresh evaluation corpus from data/ folder...")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory not found at {data_dir}. Cannot ingest corpus!")
        return

    docs = load_directory(data_dir)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP))
        
    unique_chunks = deduplicate_chunks(chunks)
    store.upsert_chunks(unique_chunks, collection_name=collection_name)
    logger.info(f"Successfully ingested {len(unique_chunks)} chunks for evaluation.")

def run_evaluation():
    logger.info("Initializing evaluation components...")
    store = QdrantStore()
    ensure_corpus_ingested(store)
    
    pipeline = AgenticQueryPipeline(store=store)
    
    if not Config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured in .env. Cannot run evaluation harness.")
        return
        
    openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    logger.info(f"Loaded {len(dataset)} evaluation questions.")
    
    results = []
    
    # Track latencies separately
    retrieval_latencies = []
    full_latencies = []
    
    for idx, item in enumerate(dataset):
        q_id = item["id"]
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        gold_answer = item["gold_answer"]
        
        logger.info(f"[{idx+1}/{len(dataset)}] Running Evaluation for query ID {q_id}: '{query}'")
        
        # Time the retrieval phase separately
        t_ret_start = time.time()
        # Retrieve using agent's internal method (which runs query analyzer first then search)
        rewritten = pipeline.analyze_query(query)
        retrieved_raw = pipeline.retrieve_context(rewritten)
        retrieval_latencies.append((time.time() - t_ret_start) * 1000)
        
        # Execute the full query pipeline
        t_full_start = time.time()
        pipeline_res = pipeline.query(query)
        full_latencies.append((time.time() - t_full_start) * 1000)
        
        # Extract returned chunk IDs
        retrieved_ids = [c["id"] for c in pipeline_res["chunks"]]
        generated_answer = pipeline_res["answer"]
        
        # 1. Compute Retrieval Metrics (evaluating the top-20 hybrid + reranked output)
        recall_1 = calculate_recall(retrieved_ids, relevant_ids, k=1)
        recall_5 = calculate_recall(retrieved_ids, relevant_ids, k=5)
        hit_rate_5 = calculate_hit_rate(retrieved_ids, relevant_ids, k=5)
        mrr = calculate_mrr(retrieved_ids, relevant_ids)
        ndcg_5 = calculate_ndcg(retrieved_ids, relevant_ids, k=5)
        context_precision_5 = calculate_context_precision(retrieved_ids, relevant_ids, k=5)
        
        # 2. Compute Match Metrics
        em = calculate_exact_match(generated_answer, gold_answer)
        f1 = calculate_f1_score(generated_answer, gold_answer)
        
        # 3. Compute LLM-as-a-judge Metrics (faithfulness, relevance)
        context_text = "\n\n".join([f"ID: {c['id']}\nText: {c['text']}" for c in pipeline_res["chunks"]])
        if not context_text:
            context_text = "Empty context. (Gate blocked)"
            
        judge_res = run_llm_judge_eval(
            openai_client=openai_client,
            query=query,
            answer=generated_answer,
            context=context_text
        )
        
        # Pack metrics
        run_data = {
            "id": q_id,
            "query": query,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "status": pipeline_res["status"],
            "retries": pipeline_res["retries"],
            "cost_usd": pipeline_res["cost"],
            "tokens_used": pipeline_res["tokens_used"],
            "latency_ms": pipeline_res["latency_ms"],
            "metrics": {
                "recall_1": recall_1,
                "recall_5": recall_5,
                "hit_rate_5": hit_rate_5,
                "mrr": mrr,
                "ndcg_5": ndcg_5,
                "context_precision_5": context_precision_5,
                "exact_match": em,
                "f1_score": f1,
                "faithfulness_score": judge_res.faithfulness_score,
                "relevance_score": judge_res.relevance_score,
            },
            "judge_explanation": judge_res.explanation
        }
        
        results.append(run_data)
        logger.info(f"Completed run. Faithfulness: {judge_res.faithfulness_score}/5 | Relevance: {judge_res.relevance_score}/5")

    # Aggregate summaries
    total_cost = sum(r["cost_usd"] for r in results)
    avg_recall_5 = np.mean([r["metrics"]["recall_5"] for r in results])
    avg_mrr = np.mean([r["metrics"]["mrr"] for r in results])
    avg_ndcg_5 = np.mean([r["metrics"]["ndcg_5"] for r in results])
    avg_precision_5 = np.mean([r["metrics"]["context_precision_5"] for r in results])
    avg_em = np.mean([r["metrics"]["exact_match"] for r in results])
    avg_f1 = np.mean([r["metrics"]["f1_score"] for r in results])
    avg_faithfulness = np.mean([r["metrics"]["faithfulness_score"] for r in results])
    avg_relevance = np.mean([r["metrics"]["relevance_score"] for r in results])
    
    # Latencies percentiles
    p50_retrieval = np.percentile(retrieval_latencies, 50)
    p95_retrieval = np.percentile(retrieval_latencies, 95)
    p50_full = np.percentile(full_latencies, 50)
    p95_full = np.percentile(full_latencies, 95)

    summary = {
        "aggregates": {
            "average_recall_5": float(avg_recall_5),
            "average_mrr": float(avg_mrr),
            "average_ndcg_5": float(avg_ndcg_5),
            "average_context_precision_5": float(avg_precision_5),
            "average_exact_match": float(avg_em),
            "average_f1_score": float(avg_f1),
            "average_faithfulness_score": float(avg_faithfulness),
            "average_relevance_score": float(avg_relevance),
            "total_cost_usd": float(total_cost),
            "average_cost_per_query_usd": float(total_cost / len(results)),
            "p50_retrieval_latency_ms": float(p50_retrieval),
            "p95_retrieval_latency_ms": float(p95_retrieval),
            "p50_full_pipeline_latency_ms": float(p50_full),
            "p95_full_pipeline_latency_ms": float(p95_full),
        },
        "runs": results
    }

    # Save to eval_results.json
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval_results.json"))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved raw evaluation results to {results_path}")

    # Generate eval_report.md
    generate_markdown_report(summary, len(dataset))

def generate_markdown_report(summary: Dict[str, Any], total_queries: int):
    aggregates = summary["aggregates"]
    
    markdown_content = f"""# Agentic RAG Evaluation Report

This report summarizes the performance, retrieval quality, response quality, latency, and costs computed for the **Cost-Efficient Agentic RAG QA Service** over a fixed dataset of `{total_queries}` test queries.

---

## 📊 Summary Metrics Table

| Metric Category | Metric Name | Score / Value | Description |
| :--- | :--- | :---: | :--- |
| **Retrieval** | Recall@5 | `{aggregates['average_recall_5']:.4f}` | Percent of relevant context chunks retrieved in top-5 |
| | Mean Reciprocal Rank (MRR) | `{aggregates['average_mrr']:.4f}` | Rank quality of the first relevant chunk |
| | nDCG@5 | `{aggregates['average_ndcg_5']:.4f}` | Normalized Discounted Cumulative Gain ranking quality |
| | Context Precision@5 | `{aggregates['average_context_precision_5']:.4f}` | Score of relevant chunks ordered correctly at top-5 |
| **Generation** | Exact Match (EM) | `{aggregates['average_exact_match']:.4%}` | Strict text match against reference gold answers |
| | F1 Score | `{aggregates['average_f1_score']:.4f}` | Word-level token overlap score |
| **LLM-as-a-Judge** | Faithfulness | `{aggregates['average_faithfulness_score']:.2f} / 5.00` | Groundedness of response based ONLY on context |
| | Answer Relevance | `{aggregates['average_relevance_score']:.2f} / 5.00` | How well the generated response answers the query |
| **Telemetry** | Total Cost (USD) | `${aggregates['total_cost_usd']:.6f}` | Combined cost for OpenAI calls during run |
| | Avg Cost per Query | `${aggregates['average_cost_per_query_usd']:.6f}` | Average expense per execution |

---

## ⚡ Latency Performance (ms)

| Pipeline Phase | p50 (Median) | p95 (95th Percentile) |
| :--- | :---: | :---: |
| **Retrieval Only** (Analysis + Hybrid Search) | `{aggregates['p50_retrieval_latency_ms']:.2f} ms` | `{aggregates['p95_retrieval_latency_ms']:.2f} ms` |
| **Full Pipeline** (Retrieval + Rerank + Gen + Critic) | `{aggregates['p50_full_pipeline_latency_ms']:.2f} ms` | `{aggregates['p95_full_pipeline_latency_ms']:.2f} ms` |

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
"""
    
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval_report.md"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info(f"Saved evaluation markdown report to {report_path}")

if __name__ == "__main__":
    run_evaluation()
