import uuid

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

import markdown
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        # Local embedding model (NO OpenAI)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.qdrant_client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.collection_name = "physical-ai-book"
        self.vector_size = 384  # MiniLM embedding size

    async def initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections = await self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def _extract_headings(self, text: str) -> List[str]:
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, "html.parser")
        return [
            h.get_text().strip()
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ]

    def chunk_text(
        self,
        text: str,
        min_chunk_size: int = 300,
        max_chunk_size: int = 600
    ) -> List[Dict[str, Any]]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []

        for para in paragraphs:
            current_chunk.append(para)
            if len(" ".join(current_chunk).split()) >= min_chunk_size:
                chunks.append({
                    "text": "\n\n".join(current_chunk).strip()
                })
                current_chunk = []

        if current_chunk:
            chunks.append({
                "text": "\n\n".join(current_chunk).strip()
            })

        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate LOCAL embeddings (no API calls)."""
        return self.embedder.encode(texts).tolist()

    async def read_markdown_files(
        self,
        docs_dir: str = "../docusaurus/docs"
    ) -> List[Dict[str, Any]]:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs directory does not exist: {docs_dir}")

        documents = []

        for file_path in docs_path.rglob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                page = str(file_path.relative_to(docs_path)).replace("\\", "/")
                headings = self._extract_headings(content)

                documents.append({
                    "page": page,
                    "content": content,
                    "headings": headings
                })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        return documents

    async def process_document(
        self,
        document: Dict[str, Any],
        chunk_id_base: str
    ) -> List[PointStruct]:

        text_chunks = self.chunk_text(document["content"])
        texts = [c["text"] for c in text_chunks]

        vectors = self.embed_texts(texts)

        points = []
        for idx, vector in enumerate(vectors):
            chunk_id = str(uuid.uuid4())

            payload = {
                "page": document["page"],
                "heading": document["headings"][0] if document["headings"] else "",
                "text": texts[idx],  # Store the text content for retrieval
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": len(vectors),
            }

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload
                )
            )

        return points

    async def ingest_documents(self) -> Dict[str, Any]:
        try:
            await self.initialize_collection()
            documents = await self.read_markdown_files()

            total_vectors = 0

            for idx, document in enumerate(documents):
                points = await self.process_document(document, f"doc_{idx}")

                if points:
                    await self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_vectors += len(points)

            return {
                "status": "success",
                "documents_processed": len(documents),
                "vectors_stored": total_vectors,
                "collection": self.collection_name
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise
