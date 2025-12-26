"""
Ingestion module for the RAG system.
Handles ingesting Docusaurus textbook markdown files into Qdrant vector database.
"""

from .service import IngestionService
from .api import router as ingestion_router

__all__ = ["IngestionService", "ingestion_router"]