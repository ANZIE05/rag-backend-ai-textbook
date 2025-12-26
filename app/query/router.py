from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from .service import RAGQueryService

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponseItem(BaseModel):
    page: str
    heading: str
    score: float
    text: str


@router.post("", response_model=List[QueryResponseItem])
async def query_rag(request: QueryRequest):
    try:
        service = RAGQueryService()
        return await service.query_rag(
            question=request.question,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health")
async def health():
    return {"status": "ok"}
