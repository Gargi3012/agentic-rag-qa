import math
import logging
from typing import List, Set, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from app.config import Config
from app.generation.agent import call_llm_with_backoff

logger = logging.getLogger("agentic_rag.eval.metrics")

# LLM-as-a-judge response schema
class JudgeEvaluation(BaseModel):
    faithfulness_score: int = Field(
        ..., 
        description="Score from 1 to 5. 5 means the generated answer is strictly and fully supported by the retrieved context. For each factual claim, there must be a matching quote in the context. If the answer is 'insufficient context' and the context is empty, return 5. If the answer makes claims not present in the context, deduct points."
    )
    relevance_score: int = Field(
        ..., 
        description="Score from 1 to 5. 5 means the answer directly and fully addresses the user query. If the answer is 'insufficient context' and the context indeed has no information to answer the query, return 5. If the context has information but the system refuses, return 1."
    )
    supporting_quotes: List[str] = Field(
        ...,
        description="List of exact sentence-level quotes extracted from the retrieved context that support the factual claims in the generated answer. Empty list if answer is 'insufficient context'."
    )
    explanation: str = Field(..., description="Justification of the scores assigned, referencing specific sentences and quotes.")

# 1. Retrieval Metrics

def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@k = (Relevant chunks retrieved in top-k) / (Total relevant chunks)
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 1.0
        
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(top_k_retrieved.intersection(relevant_set))
    return hits / len(relevant_set)

def calculate_hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Hit Rate = 1 if at least one relevant chunk is retrieved in top-k, else 0
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 1.0
        
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(top_k_retrieved.intersection(relevant_set))
    return 1.0 if hits > 0 else 0.0

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    MRR = 1 / rank of the first relevant retrieved chunk. Returns 0 if none retrieved.
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 1.0
        
    relevant_set = set(relevant_ids)
    for idx, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0

def calculate_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k (using binary relevance)
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 1.0
        
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    # Calculate DCG
    dcg = 0.0
    for idx, doc_id in enumerate(top_k_retrieved):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(idx + 2)
            
    # Calculate IDCG (Ideal DCG where all relevant docs are ranked at the top)
    idcg = 0.0
    ideal_hits = min(k, len(relevant_set))
    for idx in range(ideal_hits):
        idcg += 1.0 / math.log2(idx + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_context_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Context Precision = Sum(Precision@i * rel_i) / (Total relevant chunks retrieved in top-k)
    """
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 1.0
        
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    precision_at_i_sum = 0.0
    hits_in_k = 0
    
    for idx, doc_id in enumerate(top_k_retrieved):
        if doc_id in relevant_set:
            hits_in_k += 1
            precision_at_i = hits_in_k / (idx + 1)
            precision_at_i_sum += precision_at_i
            
    if hits_in_k == 0:
        return 0.0
        
    return precision_at_i_sum / hits_in_k

# 2. Traditional Text Match Metrics

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Calculates Exact Match (EM) score (1.0 if strings match after normalization, else 0.0).
    """
    def normalize(text: str) -> str:
        # Lowercase, strip whitespaces, remove punctuation
        import string
        text = text.lower().strip()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(text.split())
        
    return 1.0 if normalize(prediction) == normalize(ground_truth) else 0.0

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculates token-level F1 score.
    """
    def normalize_tokens(text: str) -> List[str]:
        import string
        text = text.lower().strip()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return text.split()
        
    pred_tokens = normalize_tokens(prediction)
    gt_tokens = normalize_tokens(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens).intersection(set(gt_tokens))
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

# 3. LLM-as-a-judge Evaluation

def run_llm_judge_eval(
    openai_client: OpenAI,
    query: str,
    answer: str,
    context: str,
    fallback_client: Optional[OpenAI] = None
) -> JudgeEvaluation:
    """
    Calls OpenAI to evaluate faithfulness and relevance scores (1-5 scale) with an explanation.
    """
    if answer.lower() == "insufficient context":
        # Refusal is inherently faithful (5/5) and relevant if query is out-of-context (5/5)
        # We will let LLM check or hardcode this to save tokens if we want,
        # but let's let LLM check or return standard scores to be thorough.
        pass

    system_instruction = (
        "You are an expert AI Judge evaluating RAG QA system outputs. "
        "Evaluate the generated answer against the provided Context and User Query. "
        "Score the answer on two independent scales from 1 (worst) to 5 (best):\n\n"
        "1. Faithfulness (Groundedness):\n"
        "   - 5: The answer is entirely supported by the context. No external claims, extrapolations, or hallucinations. Every factual statement has an exact matching supporting sentence in the context.\n"
        "   - 3: Most claims are backed, but contains slight outside details, assumptions, or lacks exact supporting quotes.\n"
        "   - 1: Answer contains significant hallucinations, fabricated facts, or completely ignores the context.\n\n"
        "2. Answer Relevance:\n"
        "   - 5: Answer perfectly and directly answers the user's question.\n"
        "   - 3: Answer addresses the query, but is overly vague, wordy, or misses details.\n"
        "   - 1: Answer is completely off-topic or fails to answer the question entirely (e.g. refuses to answer when context is clear).\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Locate and extract exact sentence-level quotes from the retrieved context that support each factual claim in the generated answer, and list them in the 'supporting_quotes' list.\n"
        "- If the generated answer contains ANY statement that cannot be verified by a direct context quote, you MUST deduct points from the Faithfulness score.\n"
        "- If the answer is 'insufficient context' and the context indeed has NO information to answer the query, "
        "it is considered highly faithful (5) and highly relevant (5). The 'supporting_quotes' list should be empty.\n"
        "Output your evaluation strictly in the requested JSON structure."
    )

    user_content = (
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        f"Generated Answer: {answer}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

    try:
        completion = call_llm_with_backoff(
            client=openai_client,
            messages=messages,
            response_format=JudgeEvaluation,
            model=Config.LLM_MODEL,
            temperature=0.0,
            fallback_client=fallback_client
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Failed to run LLM judge evaluation: {str(e)}")
        # Return fallback on error
        return JudgeEvaluation(
            faithfulness_score=3,
            relevance_score=3,
            supporting_quotes=[],
            explanation=f"LLM Judge execution failed: {str(e)}"
        )
