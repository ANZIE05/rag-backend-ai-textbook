"""
Integration file to add ingestion endpoints to the main FastAPI application.
"""
from fastapi import FastAPI
from . import ingestion_router


def register_ingestion_endpoints(app: FastAPI):
    """
    Register ingestion endpoints with the main FastAPI application.

    Args:
        app: Main FastAPI application instance
    """
    # Mount the ingestion router under /ingest path
    app.include_router(
        ingestion_router,
        prefix="/ingest",
        tags=["ingestion"]
    )