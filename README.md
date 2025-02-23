# RAG Document Processing System

A robust Retrieval-Augmented Generation (RAG) system with advanced document processing capabilities and comprehensive testing infrastructure.

## Features

- 📄 Multi-format Document Processing
  - PDF support with text extraction
  - OCR capabilities for image-based content
  - Intelligent chunking system

- 🤖 Advanced AI Integration
  - Multi-provider support (OpenAI, Mistral, Anthropic)
  - Vector embedding generation
  - Semantic search capabilities

- 💻 User Interface
  - Streamlit web interface
  - Animated loading indicators
  - Real-time processing feedback
  - Interactive Q&A system

- ⚙️ Technical Features
  - Async document processing
  - Comprehensive test suite
  - Error handling and recovery
  - Progress tracking
  - Caching system for embeddings

## Project Structure

```
├── frontend/           # Frontend components
│   ├── main.py        # Streamlit web interface
│   ├── assets/        # Static assets
│   └── __init__.py    # Frontend package initialization
│
├── backend/           # Backend services
│   ├── api/          # API endpoints
│   │   ├── cleanup_port.py
│   │   ├── document_processor_service.py
│   │   └── __init__.py
│   ├── processors/   # Core processing modules
│   │   ├── document_processor.py
│   │   ├── pdf_processor.py
│   │   └── __init__.py
│   ├── storage/     # Storage related modules
│   │   ├── vector_store.py
│   │   ├── cache_manager.py
│   │   └── __init__.py
│   ├── tests/       # Test suite
│   │   └── test_pdf_processor.py
│   ├── __init__.py  # Backend package initialization
│   ├── config.py    # Configuration settings
│   └── rag_engine.py # RAG system core
│   └── __pycache__/ # Python cache files
│
├── .git/            # Git repository data
├── .pytest_cache/   # Pytest cache directory
├── .replit          # Replit configuration
├── .streamlit/      # Streamlit configuration
│   └── config.toml  # Streamlit server settings
├── pyproject.toml   # Project dependencies
└── .gitignore      # Git ignore rules
```

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
MISTRAL_API_KEY=your_key_here
```

3. Start the services:
```bash
# Start the FastAPI backend
python backend/api/document_processor_service.py

# Start the Streamlit frontend
streamlit run frontend/main.py
```

## Usage

1. Upload Documents:
   - Support for PDF and text files
   - Real-time processing feedback
   - Automatic chunking and embedding

2. Ask Questions:
   - Natural language queries
   - Context-aware responses
   - Source attribution

## Testing

Run the test suite:
```bash
pytest backend/tests/
```

Test coverage includes:
- Document processing API
- PDF processing
- Vector store operations
- Cache management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.