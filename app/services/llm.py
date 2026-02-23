import requests
from typing import List, Dict
from app.config import settings
from loguru import logger

class LLMService:
    """FREE Local LLM using Ollama"""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info(f"Connected to Ollama - Using model: {self.model}")
            else:
                logger.warning("Ollama is not responding. Make sure it's running: 'ollama serve'")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Start Ollama with: 'ollama serve'")
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context (FREE - runs locally)"""
        try:
            # Build context from chunks
            context = self._build_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Call Ollama API (LOCAL - no cost)
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000,
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()['response']
                logger.info("Generated answer successfully")
                return answer
            else:
                logger.error(f"Ollama error: {response.text}")
                return "Error generating answer. Please check Ollama is running."
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Generation timed out. Try a simpler query or smaller model."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f"[Page {chunk.get('page', 'N/A')}]"
            context_parts.append(f"{page_info}\n{chunk['text']}\n")
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the final prompt"""
        return f"""You are an expert astrophysics research assistant. Answer questions based ONLY on the provided research paper context. Be precise and cite specific information from the papers. If the context doesn't contain enough information to answer, say so.

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {query}

ANSWER (be specific and reference the papers):"""