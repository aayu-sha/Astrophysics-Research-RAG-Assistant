import PyPDF2
from pathlib import Path
from typing import List, Dict
from loguru import logger
import hashlib

class PDFProcessor:
    """Extract text from PDFs and prepare for embedding"""
    
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path: Path) -> Dict[str, any]:
        """Extract text from PDF with page numbers"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                pages_text = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append({
                            'page': page_num,
                            'text': text
                        })
                
                logger.info(f"Extracted text from {num_pages} pages")
                return {
                    'pages': pages_text,
                    'num_pages': num_pages,
                    'full_text': ' '.join([p['text'] for p in pages_text])
                }
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
    
    def generate_paper_id(self, filename: str) -> str:
        """Generate unique ID for paper"""
        return hashlib.md5(filename.encode()).hexdigest()[:12]