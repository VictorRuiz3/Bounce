import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback
from backend.api.cleanup_port import ensure_port_available
from backend.processors.document_processor import DocumentProcessor
import time
import numpy as np
from typing import List, Dict, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing documents with chunking and embeddings"
)

# Initialize document processor with error handling
try:
    processor = DocumentProcessor()
except Exception as e:
    logger.error(f"Failed to initialize document processor: {e}")
    raise

class ProcessingRequest(BaseModel):
    text: str
    source_name: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test the document processor
        processor.preprocess_text("test")
        return {"status": "healthy", "message": "Service is ready"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service is not healthy")

@app.post("/process-document")
async def process_document(request: ProcessingRequest):
    """Process a single document with full processing capabilities."""
    try:
        logger.info(f"Processing document: {request.source_name}")
        start_time = time.time()

        # Process document in chunks to handle large documents
        processed_chunks = []
        embeddings_list = []

        # Initial text preprocessing
        text = processor.preprocess_text(request.text)
        chunks = processor.chunk_text(text)

        # Process chunks in batches
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = processor.create_embeddings(batch)
                processed_chunks.extend(batch)
                embeddings_list.extend(batch_embeddings)
            except Exception as batch_error:
                logger.error(f"Error processing batch {i//batch_size}: {batch_error}")
                continue

        # Prepare response
        result = {
            "chunks": processed_chunks,
            "embeddings": np.array(embeddings_list) if embeddings_list else np.array([]),
            "stats": {
                "total_chunks": len(processed_chunks),
                "processed_chunks": len(embeddings_list),
                "total_words": sum(len(chunk.split()) for chunk in processed_chunks),
                "avg_chunk_size": sum(len(chunk.split()) for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0,
                "processing_time": time.time() - start_time
            }
        }

        # Format response
        response = {
            "chunks": [{"text": chunk} for chunk in result["chunks"]],
            "embeddings": result["embeddings"].tolist() if result["embeddings"] is not None else [],
            "processing_stats": result["stats"]
        }

        logger.info(f"Successfully processed document {request.source_name} "
                   f"in {time.time() - start_time:.2f} seconds")
        return response

    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8002))
        logger.info(f"Starting FastAPI service on port {port}...")

        # Ensure port is available
        if not ensure_port_available(port):
            raise RuntimeError(f"Could not secure port {port}")

        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise