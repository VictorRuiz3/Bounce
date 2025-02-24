# RAG Document Processing System

A powerful document processing system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI backend and Streamlit frontend.

## Features

- 📄 Document Processing
  - PDF and text file support
  - Intelligent text chunking
  - Real-time processing feedback
  - Progress tracking

- 🤖 AI Integration
  - Mistral AI powered
  - Vector embeddings for semantic search
  - Context-aware responses
  - Smart caching system

- 💻 User Interface
  - Clean Streamlit interface
  - Interactive Q&A system
  - Real-time processing feedback
  - Document management

## Quick Start

1. Install Initial Dependencies:
pip install aiohttp anthropic asyncio fastapi httpx mistralai numpy openai pandas pdf2image pillow psutil pydantic pypdf2 pytesseract pytest-mock pytest pytest-asyncio requests scikit-learn sentence-transformers streamlit trafilatura twilio urllib3 uvicorn
2. Install a Specific Version of mistralai:
pip install mistralai==0.4.2
3. Run the Backend Service, on Windows PowerShell:
$env:PYTHONPATH="."; python backend/api/document_processor_service.py
4. Run the Streamlit Frontend, on Windows PowerShell:
$env:PYTHONPATH="."; streamlit run frontend/main.py --server.address 0.0.0.0 --server.port 8501
5. Access the Application in a Web Browser:
http://localhost:8501

## Usage Guide

### Document Upload
1. Navigate to the "Document Upload" section
2. Click "Browse files" or drag & drop your documents
3. Supported formats: PDF, TXT
4. Wait for processing to complete
5. You'll see progress indicators and processing stats

### Asking Questions
1. Go to the "Ask Questions" section
2. Type your question in the input box
3. Click "Get Answer"
4. View the response and relevant source contexts

### System Status
- View loaded documents in the sidebar
- Monitor processing status
- Clear all documents if needed

## Technical Details

The system runs two services:

1. FastAPI Backend (Port 8002)
   - Handles document processing
   - Manages embeddings
   - Provides RAG capabilities

2. Streamlit Frontend (Port 8501)
   - User interface
   - Document upload
   - Q&A interface

## Configuration

The system uses Mistral AI for:
- Text embeddings (mistral-embed)
- Language model responses (mistral-large-latest)

Default settings:
- Chunk size: 2000 tokens
- Chunk overlap: 400 tokens
- Vector dimension: 1024
- Max batch size: 5

## Project Structure
```
├── frontend/           # Frontend components
│   └── main.py        # Streamlit interface
├── backend/           
│   ├── api/           # API endpoints
│   ├── processors/    # Document processing
│   ├── storage/       # Vector storage
│   ├── config.py      # Configuration
│   └── rag_engine.py  # RAG system core
```

## Troubleshooting

1. If the interface doesn't load:
   - Check if both services are running

2. If document processing fails:
   - Check file format (PDF/TXT only)
   - Try with a smaller document first

3. If questions aren't answered:
   - Ensure documents are uploaded
   - Check if processing completed

## Performance Tips

- Large PDF files (>100 pages) are processed in sections
- Multiple documents can be processed in parallel
- Results are cached for faster responses
- Clear cache if memory usage is high

## System Requirements

The system runs automatically on Replit with:
- Python 3.11+
- Required packages from pyproject.toml
- Mistral AI API integration
