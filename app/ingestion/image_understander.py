"""
image_understander.py
=====================
Extracts semantic descriptions from embedded PDF images using multimodal LLMs.

Strategy:
  Primary : OpenAI GPT-4o-mini Vision          (if OPENAI_API_KEY is set)
  Fallback: Groq llama-3.2-90b-vision-preview  (if GROQ_API_KEY is set)

Returns structured JSON describing the image — used to create searchable
image chunks in the vector DB so users can ask questions about charts/figures.
"""

import os
import json
import logging
import base64
from typing import Optional, Dict, Any, List
from openai import OpenAI

logger = logging.getLogger("agentic_rag.ingestion.image_understander")

# Groq vision model that supports image inputs
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_vision_messages(image_b64: str, image_ext: str) -> List[Dict]:
    """Builds the messages payload for a vision call (OpenAI-compatible format)."""
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    mime_type = mime_map.get(image_ext.lower(), "image/png")
    data_url = f"data:{mime_type};base64,{image_b64}"

    prompt = (
        "You are an expert at analyzing figures, charts, and diagrams in research and technical documents.\n"
        "Analyze the image and provide:\n"
        "1. A clear factual description (2-4 sentences) of what this image shows.\n"
        "2. Key labels, entity names, or axis labels visible in the image.\n"
        "3. A trend or insight summary (1-2 sentences) if this is a chart/graph, otherwise 'N/A'.\n\n"
        "Respond ONLY in this exact JSON format, no extra text:\n"
        '{"description": "...", "labels": ["label1", "label2"], "trend_summary": "..."}'
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                },
            ],
        }
    ]


def _parse_vision_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parses raw LLM text response into a structured dict."""
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# GPT-4o-mini Vision  (Primary)
# ---------------------------------------------------------------------------

def _describe_with_gpt(image_b64: str, image_ext: str, openai_client: OpenAI) -> Optional[Dict[str, Any]]:
    """Calls GPT-4o-mini Vision API with the base64 image."""
    try:
        messages = _build_vision_messages(image_b64, image_ext)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=350,
            timeout=25.0,
        )
        raw = response.choices[0].message.content
        result = _parse_vision_response(raw)
        logger.info("GPT-4o-mini Vision: image described successfully.")
        return result
    except Exception as e:
        logger.warning(f"GPT-4o-mini Vision call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Groq Vision fallback  (llama-3.2-90b-vision-preview)
# ---------------------------------------------------------------------------

def _describe_with_groq(image_b64: str, image_ext: str, groq_client: OpenAI) -> Optional[Dict[str, Any]]:
    """
    Calls Groq's vision-capable Llama model as fallback.
    groq_client is an OpenAI-compatible client pointed at Groq's base URL.
    """
    try:
        messages = _build_vision_messages(image_b64, image_ext)
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            max_tokens=350,
            timeout=25.0,
        )
        raw = response.choices[0].message.content
        result = _parse_vision_response(raw)
        logger.info("Groq Vision (llama-3.2-90b): image described successfully.")
        return result
    except Exception as e:
        logger.warning(f"Groq Vision fallback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_image(
    image_b64: str,
    image_ext: str,
    openai_client: Optional[OpenAI] = None,
    groq_client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Describes an image using available multimodal LLM.
    Priority: GPT-4o-mini Vision → Groq Vision → plain fallback.

    Args:
        image_b64:     Base64-encoded image bytes (string).
        image_ext:     File extension of the image (e.g., 'png', 'jpg').
        openai_client: Initialized OpenAI client with OPENAI_API_KEY.
        groq_client:   Initialized OpenAI-compatible Groq client.

    Returns:
        dict with keys:
            description   (str)       — factual description of the image
            labels        (list[str]) — key visible labels / entities
            trend_summary (str)       — trend insight for charts, else 'N/A'
    """
    # 1. Try GPT-4o-mini Vision (most accurate)
    if openai_client:
        result = _describe_with_gpt(image_b64, image_ext, openai_client)
        if result:
            return result

    # 2. Try Groq Vision as fallback
    if groq_client:
        result = _describe_with_groq(image_b64, image_ext, groq_client)
        if result:
            return result

    # 3. Plain fallback — no vision model available
    logger.warning(
        "No vision model available (set OPENAI_API_KEY or GROQ_API_KEY). "
        "Returning placeholder image description."
    )
    return {
        "description": "Image content could not be analysed (no vision model configured).",
        "labels": [],
        "trend_summary": "N/A",
    }


def build_image_document_text(image_metadata: Dict[str, Any], vision_result: Dict[str, Any]) -> str:
    """
    Converts a vision result into searchable text that gets embedded and stored in Qdrant.

    Args:
        image_metadata: The metadata dict from the image Document object.
        vision_result:  The dict returned by describe_image().

    Returns:
        A formatted string suitable for embedding and retrieval.
    """
    page = image_metadata.get("image_page", "?")
    img_idx = image_metadata.get("image_index", "?")
    filename = image_metadata.get("filename", "unknown")
    labels_str = ", ".join(vision_result.get("labels", [])) or "none detected"
    description = vision_result.get("description", "No description available.")
    trend = vision_result.get("trend_summary", "N/A")

    return (
        f"[FIGURE — {filename}, Page {page}, Figure {img_idx}]\n\n"
        f"Description: {description}\n\n"
        f"Labels / Entities: {labels_str}\n\n"
        f"Trend / Insight: {trend}"
    )


# ---------------------------------------------------------------------------
# Convenience: build clients from env vars (used by ingestion pipeline)
# ---------------------------------------------------------------------------

def get_vision_clients():
    """
    Returns (openai_client, groq_client) initialized from environment variables.
    Either can be None if the respective key is not set.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    openai_client = None
    groq_client = None

    if openai_key and not openai_key.startswith("your_"):
        openai_client = OpenAI(api_key=openai_key)
        logger.info("OpenAI client initialized for image understanding.")

    if groq_key:
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
        )
        logger.info("Groq client initialized for image understanding fallback.")

    return openai_client, groq_client
