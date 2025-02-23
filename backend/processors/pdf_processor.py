import io
import logging
import os
from typing import Optional, List
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pypdf,
            self._extract_with_ocr
        ]

    def _extract_with_pypdf(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using PyPDF2."""
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Try direct text extraction
                    page_text = page.extract_text()

                    # If no text found, try extracting from form annotations
                    if not page_text.strip() and '/Annots' in page:
                        for annot in page['/Annots']:
                            obj = annot.get_object()
                            if '/Contents' in obj:
                                page_text += obj['/Contents'] + "\n"

                    # Add page text without slide formatting
                    if page_text.strip():
                        text += page_text + "\n"
                        logger.info(f"Extracted text from page {page_num + 1} using PyPDF2")
                    else:
                        logger.info(f"No text found on page {page_num + 1}, will try OCR")

                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} using PyPDF2: {str(e)}")
                    continue

            return text.strip() if text.strip() else None

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return None

    def _extract_with_ocr(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using OCR."""
        try:
            # Convert PDF to images with higher DPI for better OCR
            images = convert_from_bytes(pdf_bytes, dpi=300)
            text = ""

            for i, image in enumerate(images):
                try:
                    # Enhance image for better OCR
                    enhanced_image = self._enhance_image_for_ocr(image)

                    # Perform OCR on each page with improved configuration
                    page_text = pytesseract.image_to_string(
                        enhanced_image,
                        config='--psm 6'  # Assume uniform text block
                    )

                    # Add page text if not empty
                    if page_text.strip():
                        text += page_text + "\n"
                        logger.info(f"Extracted text from page {i + 1} using OCR")

                except Exception as e:
                    logger.warning(f"Failed to perform OCR on page {i + 1}: {str(e)}")
                    continue

            return text.strip() if text.strip() else None

        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return None

    def _enhance_image_for_ocr(self, image: Image) -> Image:
        """Enhance image quality for better OCR results."""
        try:
            # Convert to grayscale
            enhanced = image.convert('L')

            # Increase contrast using numpy array operations
            enhanced_array = np.array(enhanced)
            enhanced_array = 255 * (enhanced_array > 128).astype('uint8')
            enhanced = Image.fromarray(enhanced_array)

            return enhanced
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using multiple methods."""
        extracted_text = None

        # Try each extraction method until one succeeds
        for method in self.extraction_methods:
            try:
                extracted_text = method(pdf_bytes)
                if extracted_text:
                    logger.info(f"Successfully extracted text using {method.__name__}")
                    break
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {str(e)}")
                continue

        if not extracted_text:
            raise ValueError("Failed to extract text from PDF using any available method")

        return extracted_text