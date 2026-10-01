import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Config
from app.services.rag_service import RAGService
from app.dependencies import set_rag_service
from app.routers.rag import router as rag_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Initializes RAG service on startup
    """

    # Startup
    logger.info("Starting up RAG API service...")

    # Fail closed: refuse to start without an API key configured. Without this,
    # an unset APP_API_KEY would leave /query fully open to the internet. This
    # is the primary guard; verify_api_key re-checks at request time as
    # defense-in-depth in case the key is cleared after startup.
    if not Config.APP_API_KEY:
        logger.critical(
            "FATAL: APP_API_KEY is not set. Refusing to start an internet-exposed "
            "service with an unauthenticated /query endpoint."
        )
        raise RuntimeError("APP_API_KEY must be set; refusing to start")

    try:
        rag_service = RAGService()
        set_rag_service(rag_service)
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG API service...")


# Interactive docs (Swagger/ReDoc) and the OpenAPI schema are recon surface for
# an internet-exposed service. Expose them only when DEBUG is on; otherwise the
# endpoints return 404.
_docs_enabled = Config.DEBUG

# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# CORS: the previous config paired allow_origins=["*"] with
# allow_credentials=True, which makes Starlette reflect ANY origin back WITH
# credentials -> universal CSRF / cross-origin credential read. This is an
# internal service-to-service tool with no browser/cookie flow, so we:
#   - read an explicit allow-list from CORS_ALLOW_ORIGINS (comma-separated),
#     defaulting to [] (no cross-origin access at all), and
#   - keep allow_credentials=False, so a wildcard (if ever set) can never be
#     combined with credentials.
_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "")
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(rag_router, prefix=Config.API_PREFIX)


# Root endpoint (optional, for testing)
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint

    Returns:
        Basic API information
    """
    endpoints = {
        "health": f"{Config.API_PREFIX}/health",
        "query": f"{Config.API_PREFIX}/query (POST)",
    }
    # Only advertise docs when they are actually mounted (DEBUG).
    if _docs_enabled:
        endpoints["docs"] = "/docs"

    return {
        "message": "RAG Retrieval API",
        "version": Config.API_VERSION,
        "endpoints": endpoints,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
