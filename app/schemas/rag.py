from pydantic import BaseModel, Field
from typing import List, Optional

from app.core.config import Config


class QueryRequest(BaseModel):
    """Request model for query endpoint"""

    # max_length caps token/cost amplification per request (cost-DoS guard).
    # Bounded at the schema layer so over-length payloads are rejected with 422
    # before any retrieval/LLM work is triggered.
    query: str = Field(
        ...,
        description="The query string to search for",
        min_length=1,
        max_length=Config.MAX_QUERY_LENGTH,
    )

    class Config:
        json_schema_extra = {
            "example": {"query": "How long is the warranty on the X200 router?"}
        }


class NodeResponse(BaseModel):
    """Response model for a single retrieved node"""

    text: str = Field(..., description="Content of the retrieved node")
    score: float = Field(..., description="Relevance score of the node")
    file_name: Optional[str] = Field(None, description="Name of the source file")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "All X200 routers ship with a 24-month limited warranty...",
                "score": 0.91,
                "file_name": "x200-warranty.md",
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""

    nodes: List[NodeResponse] = Field(..., description="List of retrieved nodes")
    total: int = Field(..., description="Total number of nodes returned")
    processing_time: float = Field(..., description="Processing time in seconds")
    query: str = Field(..., description="Original query string")

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "text": "All X200 routers ship with a 24-month limited warranty...",
                        "score": 0.91,
                        "file_name": "x200-warranty.md",
                    }
                ],
                "total": 1,
                "processing_time": 3.45,
                "query": "How long is the warranty on the X200 router?",
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(..., description="Service health status")
    indexes_loaded: int = Field(..., description="Number of indexes loaded")
    # index_names is recon-sensitive (reveals internal index topology to anyone
    # who can hit the unauthenticated liveness probe). It is Optional and omitted
    # from the public response by default; the router only populates it when
    # DEBUG is enabled. exclude_none keeps it out of the serialized payload.
    index_names: Optional[List[str]] = Field(
        default=None,
        description="List of loaded index names (only exposed when DEBUG is enabled)",
    )

    class Config:
        json_schema_extra = {"example": {"status": "healthy", "indexes_loaded": 3}}


class ErrorResponse(BaseModel):
    """Response model for errors"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Retrieval failed",
                "detail": "Connection timeout to LlamaCloud",
            }
        }
