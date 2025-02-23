import streamlit as st
import pandas as pd
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import os
import sys

# Add project root to PYTHONPATH if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.processors.document_processor import DocumentProcessor
from backend.storage.vector_store import VectorStore 
from backend.rag_engine import RAGEngine
from backend.config import VECTOR_DIMENSION, MISTRAL_API_KEY
from backend.processors.pdf_processor import PDFProcessor

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables."""
    try:
        if 'processor' not in st.session_state:
            st.session_state.processor = DocumentProcessor()
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStore(VECTOR_DIMENSION)
        if 'rag_engine' not in st.session_state:
            if not MISTRAL_API_KEY:
                st.error("Mistral API key is not set. Please set your API key in the environment variables.")
                return False
            try:
                st.session_state.rag_engine = RAGEngine(
                    st.session_state.processor,
                    st.session_state.vector_store
                )
            except ValueError as e:
                st.error(str(e))
                return False
        if 'pdf_processor' not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor()
        if 'processing_queue' not in st.session_state:
            st.session_state.processing_queue = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = set()
        if 'batch_processing' not in st.session_state:
            st.session_state.batch_processing = False
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Upload"
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        st.error(f"Error initializing application: {str(e)}")
        return False

def set_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF6B6B;
        }
        .uploaded-files {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .css-1d391kg {
            padding-top: 3rem;
        }
        .st-emotion-cache-1wrcr25 {
            background-color: #262730;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def process_document_api_cached(text: str, source_name: str):
    """Process single document using FastAPI endpoint with caching and retries."""

    # Configure retry strategy with more attempts and shorter intervals
    retry_strategy = Retry(
        total=10,  # increased number of retries
        backoff_factor=0.1,  # shorter wait times between retries
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"]  # Allow retries on both GET and POST
    )

    # Create session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    api_url = "http://127.0.0.1:8002"  # Using 127.0.0.1 instead of 0.0.0.0

    try:
        # Wait for FastAPI server to be ready
        logger.info("Checking FastAPI service availability...")
        for attempt in range(10):  # More attempts with shorter waits
            try:
                health_check = session.get(f"{api_url}/health", timeout=2)
                if health_check.status_code == 200:
                    logger.info("FastAPI service is healthy")
                    break
                logger.warning(f"Health check attempt {attempt + 1} failed with status {health_check.status_code}")
                time.sleep(0.5)  # Shorter sleep between attempts
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check attempt {attempt + 1} failed: {str(e)}")
                if attempt == 9:  # Last attempt
                    raise Exception("Document processing service is not available")
                time.sleep(0.5)

        # Process document
        logger.info(f"Processing document: {source_name}")
        response = session.post(
            f"{api_url}/process-document",
            json={"text": text, "source_name": source_name},
            timeout=300  # 5 minutes timeout for large documents
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")

        return response.json()
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process document: {str(e)}")
    finally:
        session.close()

def read_file(uploaded_file):
    """Read uploaded file based on its type."""
    try:
        if uploaded_file.type == "application/pdf":
            return read_pdf(uploaded_file)
        else:
            return uploaded_file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def read_pdf(file):
    """Read and extract text from PDF file."""
    try:
        logger.info(f"Processing PDF file: {file.name}")
        pdf_bytes = file.getvalue()
        text = st.session_state.pdf_processor.extract_text(pdf_bytes)

        if not text:
            raise ValueError("No text could be extracted from the PDF")

        logger.info(f"Successfully extracted text from PDF: {file.name}")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        if "stream" in str(e).lower():
            st.error("Error: This PDF file appears to be corrupted or in an unsupported format.")
        elif "password" in str(e).lower():
            st.error("Error: This PDF file is password-protected. Please provide an unprotected file.")
        else:
            st.error(f"Error reading PDF: {str(e)}")
        return None

def format_time(seconds):
    """Format time duration in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def process_uploaded_files(uploaded_files):
    """Process the uploaded files using synchronous API calls."""
    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
    if not new_files:
        return

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    files_processed = 0

    for file in new_files:
        text = read_file(file)
        if text:
            try:
                with st.status(f"📄 Processing {file.name}...", expanded=True) as status:
                    total_words = len(text.split())
                    status.markdown(f"""
                    - Document: {file.name}
                    - Size: {total_words:,} words
                    - Estimated time: {format_time(total_words / 2000)}
                    """)

                    # Show processing spinner
                    with st.spinner('🔄 Processing document...'):
                        # Process document
                        result = process_document_api_cached(text, file.name)

                    # Show embedding spinner
                    with st.spinner('🧠 Adding to knowledge base...'):
                        # Update vector store
                        st.session_state.vector_store.add_documents(
                            [chunk["text"] for chunk in result["chunks"]],
                            result["embeddings"],
                            file.name
                        )

                    # Display processing stats
                    stats = result["processing_stats"]
                    status.markdown(f"""
                    ✅ Processing complete!
                    - Time taken: {format_time(stats['processing_time'])}
                    - Chunks created: {stats['total_chunks']:,}
                    - Average chunk size: {int(stats['avg_chunk_size']):,} words
                    """)

                    # Update progress bar
                    files_processed += 1
                    progress = files_processed / len(new_files)
                    progress_bar.progress(progress)

                    if progress == 1.0:
                        progress_placeholder.empty()

                    st.session_state.uploaded_files.add(file.name)

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)

def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.title("AI Document Assistant")
        st.markdown("---")

        # Navigation
        st.subheader("Navigation")
        if st.button("📄 Document Upload", use_container_width=True, key="nav_upload"):
            st.session_state.current_tab = "Upload"
        if st.button("❓ Ask Questions", use_container_width=True, key="nav_ask"):
            st.session_state.current_tab = "Ask"

        st.markdown("---")

        # Settings
        st.subheader("Settings")
        st.session_state.batch_processing = st.toggle(
            "Enable Batch Processing",
            value=st.session_state.get('batch_processing', False),
            help="Process multiple documents in parallel"
        )

        # System Status
        st.markdown("---")
        st.subheader("System Status")
        st.info(f"Documents Loaded: {len(st.session_state.uploaded_files)}")

        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.uploaded_files.clear()
            st.success("System cleared successfully!")
            st.experimental_rerun()

def render_upload_section():
    """Render the document upload section."""
    st.header("📄 Document Upload")
    st.write("Upload your documents and let AI process them for intelligent querying!")

    uploaded_files = st.file_uploader(
        "Drop your documents here",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload PDFs or text files"
    )

    if uploaded_files:
        with st.expander("📋 Processing Details", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### Document Processing Queue")
                for file in uploaded_files:
                    if file.name not in st.session_state.uploaded_files:
                        st.info(f"🔄 Processing: {file.name}")
                    else:
                        st.success(f"✅ Processed: {file.name}")
            with col2:
                st.markdown("### Stats")
                st.metric("Files Queued", len([f for f in uploaded_files if f.name not in st.session_state.uploaded_files]))
                st.metric("Files Processed", len(st.session_state.uploaded_files))

        if uploaded_files:
            process_uploaded_files(uploaded_files)

def render_qa_section():
    """Render the Q&A section."""
    st.header("❓ Ask Questions")

    if not st.session_state.vector_store.chunks:
        st.warning("⚠️ Please upload at least one document before asking questions.")
        return

    # Display loaded documents
    with st.expander("📚 Available Documents", expanded=False):
        for doc in st.session_state.uploaded_files:
            st.write(f"📄 {doc}")

    # Question input
    query = st.text_input(
        "What would you like to know about your documents?",
        placeholder="Enter your question here...",
        key="query_input"
    )

    if st.button("🔍 Get Answer", use_container_width=True):
        if not query:
            st.warning("Please enter a question.")
            return

        # Show animated progress during processing
        with st.spinner("🤔 Analyzing documents..."):
            try:
                result = st.session_state.rag_engine.process_query(query)

                # Display response in a nice format
                st.markdown("### 📝 Answer")
                st.markdown(result["response"])

                # Show relevant contexts in an expander
                with st.expander("🔍 Relevant Contexts", expanded=False):
                    for idx, chunk in enumerate(result["context"], 1):
                        with st.container():
                            st.markdown(f"**Source {idx}:** {chunk['source']}")
                            st.markdown(f"**Relevance Score:** {chunk['score']:.2f}")
                            st.markdown("**Context:**")
                            st.markdown(f">{chunk['text']}")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                logger.error(f"Error generating answer: {str(e)}", exc_info=True)

def main():
    """Main entry point for the Streamlit application."""
    set_page_config()

    if not initialize_session_state():
        st.stop()

    render_sidebar()

    # Main content based on current tab
    if st.session_state.current_tab == "Upload":
        render_upload_section()
    else:
        render_qa_section()

if __name__ == "__main__":
    main()