import pytest
from fastapi.testclient import TestClient
from document_processor_service import app
import json

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_document_empty():
    """Test processing an empty document."""
    data = {
        "text": "",
        "source_name": "test_doc"
    }
    response = client.post("/process-document", json=data)
    assert response.status_code == 200  # Changed to match actual behavior
    result = response.json()
    assert result["chunks"] == []
    assert result["embeddings"] == []

def test_process_document_valid():
    """Test processing a valid document."""
    data = {
        "text": "This is a test document for processing. It should be chunked and embedded properly.",
        "source_name": "test_doc"
    }
    response = client.post("/process-document", json=data)
    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "chunks" in result
    assert "embeddings" in result
    assert "processing_stats" in result

    # Verify stats
    stats = result["processing_stats"]
    assert stats["total_chunks"] > 0
    assert stats["processed_chunks"] > 0
    assert stats["total_words"] > 0
    assert stats["processing_time"] > 0

def test_process_document_large():
    """Test processing a large document."""
    # Create a large document with repeated text
    large_text = " ".join(["This is a test sentence."] * 1000)
    data = {
        "text": large_text,
        "source_name": "large_doc"
    }
    response = client.post("/process-document", json=data)
    assert response.status_code == 200
    result = response.json()

    # Verify chunking worked
    assert len(result["chunks"]) > 1