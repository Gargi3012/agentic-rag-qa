import logging
from fastapi import Header, HTTPException, Security, status
from app.config import Config

logger = logging.getLogger("agentic_rag.api.auth")

def get_api_key(x_api_key: str = Header(..., description="API Key for accessing endpoints")) -> str:
    """
    Dependency to validate the X-API-Key header.
    Returns the API Key if valid, otherwise raises 401 Unauthorized.
    """
    if not Config.APP_API_KEY:
        # If no key is set, log warning and allow (security disabled)
        logger.warning("APP_API_KEY is not configured in .env. Security is disabled.")
        return x_api_key
        
    if x_api_key != Config.APP_API_KEY:
        logger.warning(f"Unauthorized access attempt with invalid API key: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. Access Denied."
        )
    return x_api_key
