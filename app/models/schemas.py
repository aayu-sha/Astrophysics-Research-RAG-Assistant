from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PaperUploadResponse(BaseModel):
    paper_id: str
    filename: str
    num_chunks: int
    message: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

class SourceChunk(BaseModel):
    content: str
    page: Optional[int] = None
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    processing_time: float

class PaperMetadata(BaseModel):
    paper_id: str
    filename: str
    upload_date: datetime
    num_pages: Optional[int] = None
    num_chunks: int