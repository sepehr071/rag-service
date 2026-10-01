import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import Config
from app.dependencies import get_rag_service, verify_api_key
from app.schemas.rag import (
    ErrorResponse,
    HealthResponse,
    NodeResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["RAG"])


# ---------------------------------------------------------------------------
# Lightweight in-process rate limiter
#
# This is a BEST-EFFORT, PER-WORKER fixed-window limiter. It does NOT share
# state across processes/replicas, so the effective global limit scales with
# the number of workers. The AUTHORITATIVE rate limit must live at the reverse
# proxy / API gateway in front of this service. This guard exists only to blunt
# accidental loops and trivial abuse against a single worker without pulling in
# a new dependency. State is kept asyncio-safe via a single lock.
#
# RATE_LIMIT_PER_MINUTE is read directly from the environment (config.py is
# owned by a parallel agent); set it to 0 to disable the in-process guard.
# ---------------------------------------------------------------------------
_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_RATE_WINDOW_SECONDS: int = 60

# client_id -> (window_start_epoch, request_count)
_rate_state: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
_rate_lock = asyncio.Lock()


def _client_id(request: Request) -> str:
    """
    Identify the caller for rate limiting.

    Prefer the API key (stable per consumer) and fall back to the peer IP. We
    only read the direct socket peer here; X-Forwarded-For is spoofable unless
    a trusted proxy sets it, and the proxy is where the real limit belongs.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


async def enforce_rate_limit(request: Request) -> None:
    """
    FastAPI dependency: fixed-window per-client rate limit.

    Raises 429 with a Retry-After header (seconds until the window resets) when
    the caller exceeds the configured budget. A limit of 0 disables the guard.
    """
    if _RATE_LIMIT_PER_MINUTE <= 0:
        return

    client = _client_id(request)
    now = time.monotonic()

    async with _rate_lock:
        window_start, count = _rate_state[client]
        if now - window_start >= _RATE_WINDOW_SECONDS:
            # Start a fresh window for this client.
            _rate_state[client] = (now, 1)
            return

        if count >= _RATE_LIMIT_PER_MINUTE:
            retry_after = max(1, int(_RATE_WINDOW_SECONDS - (now - window_start)))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        _rate_state[client] = (window_start, count + 1)


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    summary="Health Check",
    description="Liveness probe for the RAG service",
)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """
    Health check endpoint to verify service status.

    Intentionally unauthenticated so orchestrators can probe liveness, but it
    does NOT leak index_names (recon surface) to anonymous callers. Index names
    are only attached when DEBUG is enabled.

    Returns:
        HealthResponse: Service health status (index names gated behind DEBUG).
    """
    try:
        health_status = rag_service.get_health_status()
    except Exception as e:
        correlation_id = uuid.uuid4().hex
        logger.error(
            "Health check failed (correlation_id=%s): %s",
            correlation_id,
            e,
            exc_info=True,
        )
        detail = str(e) if Config.DEBUG else "Internal server error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{detail} (correlation_id={correlation_id})",
        )

    # Strip recon-sensitive index_names from the public payload unless DEBUG.
    if not Config.DEBUG:
        health_status = {k: v for k, v in health_status.items() if k != "index_names"}

    return HealthResponse(**health_status)


@router.post(
    "/query",
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
    response_model=QueryResponse,
    summary="Query RAG System",
    description="Execute a query against the RAG system and retrieve relevant nodes",
    responses={
        200: {"description": "Successful query with retrieved nodes"},
        400: {"description": "Invalid query request", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Execute a retrieval query against the RAG system.

    Args:
        request: QueryRequest with the query string.

    Returns:
        QueryResponse: Retrieved nodes with metadata and scores.
    """
    # Length bounds are enforced by the schema (min_length/max_length); here we
    # only reject whitespace-only payloads that slip past min_length=1.
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        start_time = time.time()

        # Avoid logging full user query content at INFO; log only its length to
        # prevent PII / sensitive prompt leakage into shared log sinks.
        logger.info("Processing query (length=%d)", len(query))
        nodes = await rag_service.retrieve(query)

        processing_time = time.time() - start_time

        node_responses = []
        for node in nodes:
            metadata_dict = node.metadata or {}
            file_name = metadata_dict.get("file_name")

            node_responses.append(
                NodeResponse(
                    text=node.get_content(),
                    score=node.score if node.score is not None else 0.0,
                    file_name=file_name,
                )
            )

        logger.info(
            "Query completed in %.2fs, returned %d nodes",
            processing_time,
            len(node_responses),
        )

        return QueryResponse(
            nodes=node_responses,
            total=len(node_responses),
            processing_time=round(processing_time, 2),
            query=query,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Never echo internal exception text to the client by default: it can
        # leak stack details, paths, upstream provider errors, or secrets.
        # Surface a generic message + correlation id; the full exception is
        # logged server-side so operators can join on the id.
        correlation_id = uuid.uuid4().hex
        logger.error(
            "Query failed (correlation_id=%s): %s",
            correlation_id,
            e,
            exc_info=True,
        )
        detail = (
            f"Query processing failed: {e}" if Config.DEBUG else "Internal server error"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{detail} (correlation_id={correlation_id})",
        )
