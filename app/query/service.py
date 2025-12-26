import os
import logging
from typing import List, Dict, Any

import requests
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGQueryService:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.qdrant_url = os.getenv("QDRANT_URL").rstrip("/")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "physical-ai-book"

    def embed_text(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    async def query_rag(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embed_text(question)

        url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"

        headers = {
            "Content-Type": "application/json",
        }
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key

        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        results = response.json()["result"]

        formatted = []
        for r in results:
            formatted.append({
                "page": r["payload"].get("page", ""),
                "heading": r["payload"].get("heading", ""),
                "score": r["score"],
                "text": r["payload"].get("content", "")
            })

        return formatted
