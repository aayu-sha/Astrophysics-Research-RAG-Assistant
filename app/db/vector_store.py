import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from loguru import logger
from app.config import settings

class VectorStore:
    """FAISS-based vector storage and retrieval"""
    
    def __init__(self, dimension: int = None):
        self.index = None
        self.chunks_metadata = []
        self.dimension = dimension
        
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Create FAISS index from embeddings"""
        try:
            # Get dimension from embeddings
            if self.dimension is None:
                self.dimension = embeddings.shape[1]
            
            # Create FAISS index (L2 distance)
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store metadata
            self.chunks_metadata = metadata
            
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors (dim={self.dimension})")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        try:
            if self.index is None:
                raise ValueError("Index not initialized. Upload a paper first.")
            
            # Reshape query embedding
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks_metadata):
                    result = self.chunks_metadata[idx].copy()
                    result['score'] = float(1 / (1 + dist))
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def save(self, path: Path = None):
        """Save index and metadata to disk"""
        try:
            save_path = path or settings.FAISS_INDEX_PATH
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(save_path / "index.faiss"))
            
            # Save metadata
            with open(save_path / "metadata.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks_metadata,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Saved index to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load(self, path: Path = None):
        """Load index and metadata from disk"""
        try:
            load_path = path or settings.FAISS_INDEX_PATH
            
            # Load FAISS index
            self.index = faiss.read_index(str(load_path / "index.faiss"))
            
            # Load metadata
            with open(load_path / "metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks_metadata = data['chunks']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded index from {load_path} (dim={self.dimension})")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise