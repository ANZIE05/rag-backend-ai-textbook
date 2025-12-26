from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from .service import IngestionService
import logging


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", summary="Ingest Docusaurus textbook into RAG system")
async def ingest_textbook():
    """
    Ingest all markdown files from the docs directory into the RAG system.

    This endpoint:
    - Reads all markdown files from ../docs directory
    - Chunks the text (500-800 tokens)
    - Generates embeddings using OpenAI
    - Stores in Qdrant collection: physical-ai-book
    - Stores metadata: page, heading, chunk_id
    """
    try:
        service = IngestionService()
        result = await service.ingest_documents()

        return JSONResponse(
            status_code=200,
            content=result
        )
    except FileNotFoundError as e:
        logger.error(f"Docs directory not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/health", summary="Check ingestion service health")
async def health_check():
    """
    Health check endpoint for the ingestion service.
    """
    return {"status": "healthy", "service": "ingestion"}