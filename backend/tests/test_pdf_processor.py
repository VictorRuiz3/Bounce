import pytest
from backend.processors.pdf_processor import PDFProcessor
import io
from PyPDF2 import PdfWriter, PdfReader
from PIL import Image
import numpy as np

@pytest.fixture
def pdf_processor():
    return PDFProcessor()

def create_test_pdf_bytes():
    """Create a simple PDF file for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()

def test_init():
    """Test PDFProcessor initialization."""
    processor = PDFProcessor()
    assert len(processor.extraction_methods) == 2

def test_preprocess_empty_pdf(pdf_processor):
    """Test processing an empty PDF."""
    pdf_bytes = create_test_pdf_bytes()
    with pytest.raises(ValueError, match="Failed to extract text from PDF"):
        pdf_processor.extract_text(pdf_bytes)

def test_enhance_image_for_ocr(pdf_processor):
    """Test image enhancement for OCR."""
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')
    enhanced = pdf_processor._enhance_image_for_ocr(test_image)

    # Check if enhanced image is grayscale
    assert enhanced.mode == 'L'

def test_extract_text_with_pypdf(pdf_processor, mocker):
    """Test text extraction using PyPDF2."""
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = "Test text content"

    mock_reader = mocker.MagicMock()
    mock_reader.pages = [mock_page]

    mocker.patch('PyPDF2.PdfReader', return_value=mock_reader)

    pdf_bytes = create_test_pdf_bytes()
    text = pdf_processor._extract_with_pypdf(pdf_bytes)
    assert text == "Test text content"

@pytest.mark.skip(reason="OCR tests require test images")
def test_extract_text_with_ocr(pdf_processor):
    """Test OCR text extraction."""
    # This test would require actual PDF with images
    # Marked as skip for now
    pass