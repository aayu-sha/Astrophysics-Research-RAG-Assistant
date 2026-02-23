from typing import List, Dict
from app.config import settings

class TextChunker:
    """Split text into chunks for embedding"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def chunk_text(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Split text into overlapping chunks while preserving page info
        """
        chunks = []
        
        for page_info in pages_data:
            text = page_info['text']
            page_num = page_info['page']
            
            # Simple word-based chunking
            words = text.split()
            
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) > 50:  # Minimum chunk size
                    chunks.append({
                        'text': chunk_text,
                        'page': page_num,
                        'chunk_id': len(chunks)
                    })
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        text = ' '.join(text.split())
        return text