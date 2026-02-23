# Astrophysics Research RAG Assistant

A fully functional RAG (Retrieval-Augmented Generation) system for astrophysics research papers.

## ✨ Features
- Upload and process PDF research papers
- Semantic search across papers
- AI-powered question answering
- Source attribution with page numbers
- 100% free and runs locally

## 🚀 Tech Stack
- **Backend:** FastAPI
- **Embeddings:** Sentence-Transformers (local)
- **Vector DB:** FAISS
- **LLM:** Ollama (Llama 3.1)
- **PDF Processing:** PyPDF2

## 📦 Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Run
python -m uvicorn app.main:app --reload
```

## 🧪 Demo
[Screenshots or GIF here]

## 📊 Performance
- Upload: ~10-30 seconds per paper
- Query: ~10-20 seconds
- Cost: $0 (runs locally)