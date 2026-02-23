from sentence_transformers import SentenceTransformer
from typing import List
from app.config import settings
from loguru import logger
import numpy as np

class EmbeddingService:
    """Generate embeddings using FREE HuggingFace models (runs locally)"""
    
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        # This downloads the model ONCE, then runs locally (no API calls)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts (FREE - runs on your CPU/GPU)"""
        try:
            # This runs LOCALLY - no internet needed after first download
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()