from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path

from app.models.schemas import (
    PaperUploadResponse, 
    QueryRequest, 
    QueryResponse,
    SourceChunk
)
from app.services.pdf_processor import PDFProcessor
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalPipeline
from app.utils.text_processing import TextChunker
from app.db.vector_store import VectorStore
from app.config import settings
from loguru import logger

router = APIRouter()

# Initialize services
pdf_processor = PDFProcessor()
embedding_service = EmbeddingService()
text_chunker = TextChunker()
vector_store = VectorStore(dimension=embedding_service.embedding_dimension)
retrieval_pipeline = RetrievalPipeline()

@router.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    """Upload and process a research paper PDF"""
    
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate paper ID
        paper_id = pdf_processor.generate_paper_id(file.filename)
        
        # Save uploaded file
        file_path = settings.UPLOAD_DIR / f"{paper_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved PDF: {file.filename}")
        
        # Extract text
        extracted_data = pdf_processor.extract_text(file_path)
        
        # Chunk text
        chunks = text_chunker.chunk_text(extracted_data['pages'])
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts)
        
        # Add to vector store
        try:
            vector_store.load()
        except:
            pass  # No existing index
        
        # Create or update index
        metadata = [
            {
                'text': chunk['text'],
                'page': chunk['page'],
                'paper_id': paper_id,
                'filename': file.filename
            }
            for chunk in chunks
        ]
        
        vector_store.create_index(embeddings, metadata)
        vector_store.save()
        
        # Update retrieval pipeline's vector store
        retrieval_pipeline.vector_store = vector_store
        
        return PaperUploadResponse(
            paper_id=paper_id,
            filename=file.filename,
            num_chunks=len(chunks),
            message=f"Successfully processed {file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Query the research papers"""
    
    try:
        # Execute RAG pipeline
        result = retrieval_pipeline.query(request.query, top_k=request.top_k)
        
        # Format sources
        sources = [
            SourceChunk(
                content=chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                page=chunk.get('page'),
                score=chunk['score']
            )
            for chunk in result['sources']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            query=request.query,
            processing_time=result['processing_time']
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "app": settings.APP_NAME}