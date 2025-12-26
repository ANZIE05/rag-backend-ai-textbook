#!/usr/bin/env python3
"""
Simple test script to verify the ingestion service works correctly.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ingest.service import IngestionService


async def test_ingestion():
    """Test the ingestion service."""
    print("Testing ingestion service...")

    # Check if required environment variables are set
    required_env_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        print("Please set them before running the ingestion.")
        return False

    try:
        service = IngestionService()

        # Test the collection initialization
        print("Initializing collection...")
        await service.initialize_collection()

        # Test reading markdown files (this will fail if docs dir doesn't exist)
        print("Testing reading markdown files...")
        try:
            docs = await service.read_markdown_files("../docs")
            print(f"Found {len(docs)} markdown files")

            if docs:
                # Test chunking on the first document
                first_doc = docs[0]
                print(f"Testing chunking on: {first_doc['page']}")

                chunks = service.chunk_text(first_doc['content'], min_chunk_size=300, max_chunk_size=600)
                print(f"Chunked into {len(chunks)} pieces")

                for i, chunk in enumerate(chunks[:2]):  # Just test first 2 chunks
                    print(f"  Chunk {i+1}: {chunk['tokens']} tokens")

                    # Test embedding generation (but limit to avoid spending too much on API calls in test)
                    if i == 0:
                        print("Testing embedding generation...")
                        embeddings = await service.get_embeddings([chunk['text']])
                        print(f"Generated embedding with {len(embeddings[0])} dimensions")

        except FileNotFoundError as e:
            print(f"Docs directory not found: {e}")
            print("This is expected if the docs directory doesn't exist yet.")
            return True  # This is not an error in the service itself

        print("All tests passed!")
        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ingestion())
    if not success:
        sys.exit(1)