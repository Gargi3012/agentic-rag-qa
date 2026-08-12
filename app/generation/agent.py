import logging
import time
import os
import json
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


class CriticCheck(BaseModel):
    is_grounded: bool = Field(description="True if every fact in the answer is directly supported by the context. False if there are hallucinations or outside information.")
    reason: str = Field(description="Explanation of the findings, explaining any ungrounded statements.")

def call_llm_with_backoff(client: OpenAI, messages: List[Dict[str, str]], response_format=None, fallback_client: Optional[OpenAI] = None, **kwargs) -> Any:
    """
    Utility to invoke OpenAI chat completions with exponential backoff on connection/timeout errors.
    Falls back to Groq client if OpenAI fails or is not configured.
    """
    max_retries = 3
    backoff = 2.0
    
    # Try calling OpenAI
    if client:
        for attempt in range(max_retries):
            try:
                if response_format:
                    logger.info("Calling OpenAI Structured completion...")
                    return client.beta.chat.completions.parse(
                        messages=messages,
                        response_format=response_format,
                        timeout=10.0,
                        **kwargs
                    )
                else:
                    logger.info("Calling OpenAI Standard completion...")
                    return client.chat.completions.create(
                        messages=messages,
                        timeout=10.0,
                        **kwargs
                    )
            except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError) as e:
                # If OpenAI quota/rate limits are hit, try fallback immediately if available
                if fallback_client:
                    logger.warning(f"OpenAI error: {str(e)}. Falling back to Groq immediately...")
                    break
                
                if attempt == max_retries - 1:
                    logger.error(f"OpenAI API call failed after {max_retries} retries: {str(e)}")
                    raise e
                sleep_time = backoff ** (attempt + 1)
                logger.warning(f"OpenAI API error: {str(e)}. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            except Exception as e:
                # Any other OpenAI error (like quota exceeded)
                if fallback_client:
                    logger.warning(f"OpenAI error ({str(e)}). Falling back to Groq immediately...")
                    break
                logger.error(f"Unexpected OpenAI error: {str(e)}")
                raise e

    # Fallback to Groq if OpenAI failed or was skipped
    if fallback_client:
        logger.info("Executing fallback call to Groq client...")
        try:
            # Replace model with Groq Llama 3.1 8B model
            groq_model = "llama-3.1-8b-instant"
            
            if response_format:
                # Groq doesn't support beta.chat.completions.parse, so use json_object mode
                # We need to instruct the model to return JSON in the system prompt
                modified_messages = list(messages)
                json_instruction = "IMPORTANT: Return the response ONLY as a valid JSON object matching the requested schema."
                if modified_messages[0]["role"] == "system":
                    modified_messages[0] = {
                        "role": "system",
                        "content": modified_messages[0]["content"] + "\n\n" + json_instruction
                    }
                
                logger.info(f"Calling Groq with JSON format using {groq_model}...")
                response = fallback_client.chat.completions.create(
                    messages=modified_messages,
                    model=groq_model,
                    response_format={"type": "json_object"},
                    timeout=10.0
                )
                
                # Parse string content
                content = response.choices[0].message.content
                logger.info(f"Groq raw response: {content}")
                parsed_json = json.loads(content)
                parsed_obj = response_format.model_validate(parsed_json)
                
                # Mock the completion object
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
                
                class MockParsedMessage:
                    def __init__(self, obj, raw_content):
                        self.parsed = obj
                        self.content = raw_content

                class MockChoice:
                    def __init__(self, obj, raw_content):
                        self.message = MockParsedMessage(obj, raw_content)

                class MockUsage:
                    def __init__(self, p_tok, c_tok):
                        self.prompt_tokens = p_tok
                        self.completion_tokens = c_tok
                        self.total_tokens = p_tok + c_tok

                class MockCompletion:
                    def __init__(self, obj, raw_content, p_tok, c_tok):
                        self.choices = [MockChoice(obj, raw_content)]
                        self.usage = MockUsage(p_tok, c_tok)

                return MockCompletion(parsed_obj, content, prompt_tokens, completion_tokens)
            else:
                logger.info(f"Calling Groq standard completion using {groq_model}...")
                return fallback_client.chat.completions.create(
                    messages=messages,
                    model=groq_model,
                    timeout=10.0
                )
        except Exception as groq_e:
            logger.error(f"Groq fallback call failed: {str(groq_e)}")
            raise groq_e
    else:
        raise RuntimeError("No LLM client (OpenAI or Groq) succeeded or was configured.")


import re

def compress_context(chunk_text: str, query: str, max_sentences: int = 3) -> str:
    """
    Trims a text chunk to only the most relevant sentences.
    Computes a keyword overlap score between the query terms and sentences.
    """
    query_words = set(query.lower().split())
    # Basic stop words to ignore in keyword matching
    stop_words = {"what", "is", "the", "how", "does", "to", "in", "a", "an", "and", "of", "for", "on", "with", "about", "by", "why", "are", "you", "i"}
    query_keywords = query_words - stop_words
    if not query_keywords:
        query_keywords = query_words
        
    # Split text into sentences using simple regex
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    if len(sentences) <= max_sentences:
        return chunk_text
        
    scored_sentences = []
    for idx, sent in enumerate(sentences):
        # Basic word matching score
        sent_words = set(sent.lower().split())
        overlap = len(query_keywords.intersection(sent_words))
        scored_sentences.append((overlap, idx, sent))
        
    # Sort by overlap score descending, then by original index ascending
    scored_sentences.sort(key=lambda x: (-x[0], x[1]))
    
    # Select top sentences
    top_selections = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
    return " ".join(t[2] for t in top_selections)


class AgenticQueryPipeline:
    def __init__(self, store: Optional[QdrantStore] = None):
        self.store = store or QdrantStore()
        
        # Initialize OpenAI client
        if not Config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not configured. Primary OpenAI client is disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            
        # Initialize Groq fallback client
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if self.groq_api_key:
            self.fallback_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_api_key
            )
            logger.info("Groq fallback client initialized.")
        else:
            self.fallback_client = None
            logger.warning("GROQ_API_KEY is not configured. Fallback client is disabled.")

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
                temperature=0.0,
                fallback_client=self.fallback_client
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

    def rerank_and_gate(self, query: str, chunks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool, float]:
        """
        Step 3 & 4: Cross-Encoder Reranking & Relevance Gate
        Reranks top-20 chunks using local Cross-Encoder.
        If the highest score is below Config.RELEVANCE_THRESHOLD, triggers gate failure.
        """
        if not chunks:
            logger.info("No chunks to rerank. Relevance gate failed.")
            return [], False, 0.0
            
        logger.info(f"Reranking {len(chunks)} chunks using Cross-Encoder...")
        # Get top-5 reranked chunks
        reranked = rerank_chunks(query, chunks, top_k=5)
        
        if not reranked:
            logger.info("Rerank returned no documents. Relevance gate failed.")
            return [], False, 0.0
            
        best_score = reranked[0]["rerank_score"]
        logger.info(f"Reranked top chunk score: {best_score:.4f} (Threshold: {Config.RELEVANCE_THRESHOLD})")
        
        if best_score < Config.RELEVANCE_THRESHOLD:
            logger.warning(f"Best score {best_score:.4f} is below relevance threshold {Config.RELEVANCE_THRESHOLD}. Gate failed.")
            return [], False, float(best_score)
            
        logger.info(f"Relevance gate passed. Proceeding with {len(reranked)} chunks.")
        return reranked, True, float(best_score)


# Add these methods to AgenticQueryPipeline
    def generate_grounded_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        strict_mode: bool = False,
        critic_feedback: str = ""
    ) -> tuple[str, int, int]:
        """
        Calls OpenAI to generate an answer grounded in the compressed context.
        """
        if not self.client:
            return "insufficient context", 0, 0

        # Build context prompt
        context_str_list = []
        for c in chunks:
            # Compress chunk text
            compressed_text = compress_context(c["text"], query, max_sentences=3)
            context_str_list.append(
                f"--- \n"
                f"Chunk ID: {c['id']}\n"
                f"Filename: {c['metadata'].get('filename', 'unknown')}\n"
                f"Content: {compressed_text}\n"
                f"---"
            )
        context_block = "\n\n".join(context_str_list)

        system_instruction = (
            "You are a highly factual QA system. Answer the User Query based ONLY on the provided Context. "
            "For any fact you state, you MUST cite the source chunk using the exact format [cite: chunk_id] where chunk_id is the exact ID provided in the context. "
            "Do not state any fact that is not directly supported by the Context. "
            "If the provided Context is insufficient to answer the query, refuse to answer and return 'insufficient context' exactly. "
            "Do not use outside knowledge or extrapolate."
        )

        user_content = f"Context:\n{context_block}\n\nUser Query: {query}"
        
        if strict_mode:
            system_instruction += (
                "\n\nCRITICAL WARNING: Your previous attempt was flagged by the critic for not being fully grounded. "
                "You MUST adhere strictly to the context. Do NOT make claims that aren't explicit. "
                "Here is the critic feedback from your last attempt:\n"
                f"{critic_feedback}"
            )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info("Generating answer from OpenAI...")
            completion = call_llm_with_backoff(
                client=self.client,
                messages=messages,
                model=Config.LLM_MODEL,
                temperature=0.0,
                fallback_client=self.fallback_client
            )
            answer = completion.choices[0].message.content.strip()
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            return answer, prompt_tokens, completion_tokens
        except Exception as e:
            logger.error(f"Failed to generate grounded answer: {str(e)}")
            raise e

    def run_critic_pass(self, query: str, answer: str, chunks: List[Dict[str, Any]]) -> tuple[CriticCheck, int, int]:
        """
        Invokes a cheap LLM call to verify if the answer is fully grounded in the provided context chunks.
        """
        if not self.client:
            return CriticCheck(is_grounded=True, reason="No LLM client"), 0, 0

        # Build context block
        context_str_list = []
        for c in chunks:
            context_str_list.append(f"Chunk ID: {c['id']}\nText: {c['text']}")
        context_block = "\n\n".join(context_str_list)

        system_instruction = (
            "You are an AI Critic evaluating groundedness for a RAG QA system. "
            "Evaluate if the candidate answer is fully supported by and grounded in the provided Context. "
            "Check for:\n"
            "1. Statements in the answer not backed by the context (hallucinations).\n"
            "2. Failure to use the exact [cite: chunk_id] tag when stating facts.\n"
            "If the candidate answer says 'insufficient context', it is considered grounded (is_grounded = True).\n"
            "Be strict. If any part of the answer is unsupported, set is_grounded to False."
        )

        user_content = (
            f"Context:\n{context_block}\n\n"
            f"User Query: {query}\n\n"
            f"Candidate Answer: {answer}"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info("Running Critic Pass check on the candidate answer...")
            completion = call_llm_with_backoff(
                client=self.client,
                messages=messages,
                response_format=CriticCheck,
                model=Config.LLM_MODEL,
                temperature=0.0,
                fallback_client=self.fallback_client
            )
            check: CriticCheck = completion.choices[0].message.parsed
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            logger.info(f"Critic Pass result: grounded={check.is_grounded} (Reason: {check.reason})")
            return check, prompt_tokens, completion_tokens
        except Exception as e:
            logger.error(f"Critic Pass execution failed: {str(e)}")
            # Default to True in case of critic API failure to avoid infinite rejections
            return CriticCheck(is_grounded=True, reason=f"Critic API Error: {str(e)}"), 0, 0

    def query(self, query_text: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the full agentic query pipeline:
        Analyzer -> Retrieval -> Rerank & Gate -> Compression -> Generation -> Critic Pass (with 1 retry)
        """
        start_time = time.time()
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        retries_count = 0
        confidence = "high"
        
        # Edge Case: Empty query
        if len(query_text.strip()) == 0:
            return {
                "answer": "insufficient context",
                "chunks": [],
                "confidence": "none",
                "retries": 0,
                "status": "empty_query_refusal",
                "latency_ms": 0.0,
                "cost": 0.0,
                "tokens_used": 0
            }

        # 1. Query Analyzer
        rewritten_query = self.analyze_query(query_text)
        
        # 2. Hybrid Retrieval (dense + sparse top-20)
        retrieved_chunks = self.retrieve_context(rewritten_query, filter_dict)
        
        # 3 & 4. Cross-Encoder Reranking & Relevance Gate
        reranked_chunks, passed_gate, best_rerank_score = self.rerank_and_gate(rewritten_query, retrieved_chunks)
        
        if not passed_gate:
            latency = (time.time() - start_time) * 1000
            return {
                "answer": "insufficient context",
                "chunks": [],
                "confidence": "none",
                "retries": 0,
                "status": "relevance_gate_refusal",
                "latency_ms": latency,
                "cost": 0.0,
                "tokens_used": 0,
                "rerank_score": best_rerank_score
            }

        # 5 & 6. Grounded Generation
        # Initial Attempt
        answer, p_tokens, c_tokens = self.generate_grounded_answer(rewritten_query, reranked_chunks)
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        
        # If the answer is an explicit refusal, we skip the critic check
        if answer.lower() == "insufficient context":
            latency = (time.time() - start_time) * 1000
            cost = (total_prompt_tokens * 0.00000015) + (total_completion_tokens * 0.00000060)
            return {
                "answer": "insufficient context",
                "chunks": [
                    {
                        "id": c["id"],
                        "text": c["text"],
                        "metadata": c["metadata"],
                        "score": c.get("rerank_score", 0.0)
                    } for c in reranked_chunks
                ],
                "confidence": "none",
                "retries": 0,
                "status": "model_refusal",
                "latency_ms": latency,
                "cost": cost,
                "tokens_used": total_prompt_tokens + total_completion_tokens,
                "rerank_score": best_rerank_score
            }

        # 7. Critic Pass
        critic_check, cp_p_tokens, cp_c_tokens = self.run_critic_pass(rewritten_query, answer, reranked_chunks)
        total_prompt_tokens += cp_p_tokens
        total_completion_tokens += cp_c_tokens
        
        # If Critic fails, regenerate once
        if not critic_check.is_grounded:
            logger.warning("Critic pass failed. Triggering strict generation retry...")
            retries_count += 1
            
            # Second attempt with feedback
            answer, p_tokens, c_tokens = self.generate_grounded_answer(
                query=rewritten_query,
                chunks=reranked_chunks,
                strict_mode=True,
                critic_feedback=critic_check.reason
            )
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            
            # Run final critic pass on retry
            critic_check, cp_p_tokens, cp_c_tokens = self.run_critic_pass(rewritten_query, answer, reranked_chunks)
            total_prompt_tokens += cp_p_tokens
            total_completion_tokens += cp_c_tokens
            
            if not critic_check.is_grounded:
                logger.error("Critic pass failed twice. Returning best-effort answer with low-confidence flag.")
                confidence = "low"
            else:
                logger.info("Critic pass passed on retry.")

        # Compute cost
        cost = (total_prompt_tokens * 0.00000015) + (total_completion_tokens * 0.00000060)
        latency = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "chunks": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "score": c.get("rerank_score", 0.0)
                } for c in reranked_chunks
            ],
            "confidence": confidence,
            "retries": retries_count,
            "status": "success",
            "latency_ms": latency,
            "cost": cost,
            "tokens_used": total_prompt_tokens + total_completion_tokens,
            "rerank_score": best_rerank_score
        }
