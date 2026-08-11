import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import openai
from app.config import Config
from app.retrieval import QdrantStore, rerank_chunks

logger = logging.getLogger("agentic_rag.generation.agent")

# Define Pydantic schema for Query Analyzer structured output
class QueryAnalysis(BaseModel):
    needs_rewrite: bool = Field(description="Whether the original query needs expansion, clarification, or rewriting.")
    suggested_query: str = Field(description="The reformulated query for better keyword matching and semantic search, or original query if no rewrite is needed.")
    reason: str = Field(description="Explanation of why rewriting was or was not necessary.")

def call_llm_with_backoff(client: OpenAI, messages: List[Dict[str, str]], response_format=None, **kwargs) -> Any:
    """
    Utility to invoke OpenAI chat completions with exponential backoff on connection/timeout errors.
    """
    max_retries = 3
    backoff = 2.0
    
    for attempt in range(max_retries):
        try:
            if response_format:
                return client.beta.chat.completions.parse(
                    messages=messages,
                    response_format=response_format,
                    timeout=10.0,
                    **kwargs
                )
            else:
                return client.chat.completions.create(
                    messages=messages,
                    timeout=10.0,
                    **kwargs
                )
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError) as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API call failed after {max_retries} retries: {str(e)}")
                raise e
            sleep_time = backoff ** (attempt + 1)
            logger.warning(f"OpenAI API error: {str(e)}. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Unexpected OpenAI error: {str(e)}")
            raise e

class AgenticQueryPipeline:
    def __init__(self, store: Optional[QdrantStore] = None):
        self.store = store or QdrantStore()
        
        if not Config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not configured. LLM calls will fail.")
            self.client = None
        else:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def analyze_query(self, query: str) -> str:
        """
        Step 1: Query Analyzer
        Decides if the query needs rewriting/expansion for better retrieval. Capped at 1 rewrite.
        """
        if not self.client:
            logger.warning("No OpenAI client available. Skipping query analysis; using original query.")
            return query
            
        # Validate query length edge case
        if len(query.strip()) > 1000:
            logger.warning("Query length exceeds 1000 characters. Truncating to avoid token blowup.")
            query = query[:1000].strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Query Analyzer for an Agentic RAG system. "
                    "Analyze the user's input query. Determine if it needs to be rewritten or expanded "
                    "to improve dense similarity search and sparse keyword retrieval. "
                    "For example, resolve vague pronouns, add relevant synonyms, or make implicit needs explicit. "
                    "Keep the final output concise and direct."
                )
            },
            {
                "role": "user",
                "content": f"User Query: {query}"
            }
        ]

        try:
            logger.info(f"Analyzing query: '{query}'")
            completion = call_llm_with_backoff(
                client=self.client,
                messages=messages,
                response_format=QueryAnalysis,
                model=Config.LLM_MODEL,
                temperature=0.0
            )
            analysis: QueryAnalysis = completion.choices[0].message.parsed
            
            if analysis.needs_rewrite:
                logger.info(f"Query rewritten: '{query}' -> '{analysis.suggested_query}' (Reason: {analysis.reason})")
                return analysis.suggested_query
            else:
                logger.info(f"Query analysis: No rewrite needed. Reason: {analysis.reason}")
                return query
        except Exception as e:
            logger.error(f"Failed to analyze query due to error: {str(e)}. Falling back to original query.")
            return query

    def retrieve_context(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Step 2: Hybrid Retrieval
        Calls our Qdrant hybrid search, returning fused dense+sparse results, top-20.
        """
        logger.info(f"Retrieving chunks for query: '{query}'")
        try:
            # Query Qdrant for top-20 fused documents
            return self.store.hybrid_search(
                query=query,
                top_k=20,
                filter_dict=filter_dict
            )
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            # If Qdrant search fails, return empty list (handled downstream by relevance gate)
            return []

    def rerank_and_gate(self, query: str, chunks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
        """
        Step 3 & 4: Cross-Encoder Reranking & Relevance Gate
        Reranks top-20 chunks using local Cross-Encoder.
        If the highest score is below Config.RELEVANCE_THRESHOLD, triggers gate failure.
        """
        if not chunks:
            logger.info("No chunks to rerank. Relevance gate failed.")
            return [], False
            
        logger.info(f"Reranking {len(chunks)} chunks using Cross-Encoder...")
        # Get top-5 reranked chunks
        reranked = rerank_chunks(query, chunks, top_k=5)
        
        if not reranked:
            logger.info("Rerank returned no documents. Relevance gate failed.")
            return [], False
            
        best_score = reranked[0]["rerank_score"]
        logger.info(f"Reranked top chunk score: {best_score:.4f} (Threshold: {Config.RELEVANCE_THRESHOLD})")
        
        if best_score < Config.RELEVANCE_THRESHOLD:
            logger.warning(f"Best score {best_score:.4f} is below relevance threshold {Config.RELEVANCE_THRESHOLD}. Gate failed.")
            return [], False
            
        logger.info(f"Relevance gate passed. Proceeding with {len(reranked)} chunks.")
        return reranked, True
