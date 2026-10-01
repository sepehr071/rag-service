import hmac
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.services.rag_service import RAGService
from app.core.config import Config

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the caller's API key, failing closed.

    Defense-in-depth: main.py's lifespan refuses to start when APP_API_KEY is
    unset, but we re-check here so the endpoint can never be served openly even
    if the startup guard is bypassed (e.g. key cleared at runtime). A missing
    server-side key is a misconfiguration, not an invitation to serve traffic,
    so we return 503 rather than silently allowing the request.
    """
    if not Config.APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service misconfigured: API key not set",
        )
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials - Missing API Key",
        )
    # compare_digest avoids leaking the key length/prefix via response-timing.
    if not hmac.compare_digest(api_key, Config.APP_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return api_key


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Dependency to get the RAG service instance.
    Raises 503 if service is not initialized.
    """
    if _rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return _rag_service


def set_rag_service(service: RAGService):
    """Set the global RAG service instance"""
    global _rag_service
    _rag_service = service
