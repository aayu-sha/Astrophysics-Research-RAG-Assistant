from typing import List, Dict
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.db.vector_store import VectorStore
from loguru import logger
import time

class RetrievalPipeline:
    """End-to-end RAG pipeline (100% FREE)"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Initialize vector store with correct dimension
        dimension = self.embedding_service.embedding_dimension
        self.vector_store = VectorStore(dimension=dimension)
        
        # Try to load existing index
        try:
            self.vector_store.load()
            logger.info("Loaded existing vector store")
        except:
            logger.info("No existing vector store found")
    
    def query(self, query_text: str, top_k: int = 5) -> Dict:
        """Execute RAG query (FREE - no API costs)"""
        start_time = time.time()
        
        try:
            # 1. Generate query embedding (runs locally)
            logger.info("Generating query embedding...")
            query_embedding = self.embedding_service.generate_single_embedding(query_text)
            
            # 2. Search vector store
            logger.info("Searching vector database...")
            retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)
            
            # 3. Generate answer (runs locally with Ollama)
            logger.info("Generating answer with local LLM...")
            answer = self.llm_service.generate_answer(query_text, retrieved_chunks)
            
            processing_time = time.time() - start_time
            
            return {
                'answer': answer,
                'sources': retrieved_chunks,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval pipeline: {e}")
            raise