from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings
from loguru import logger
import sys

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG system for astrophysics research papers",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["papers"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.APP_NAME}")
    logger.info(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"🗄️  Vector store: {settings.FAISS_INDEX_PATH}")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down application")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)