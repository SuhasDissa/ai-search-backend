from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import os
import shutil
from .config import settings
from .model_handler import model_handler
from .rag_engine import rag_engine
from .web_search import web_search

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load models on startup
    print("Starting AI Spotlight Backend...")
    model_handler.load_model()

    # Try to load existing RAG index
    if os.path.exists(f"{settings.FAISS_INDEX_PATH}.index"):
        rag_engine.load_index()
    else:
        print("No existing RAG index found. Upload documents to create one.")
    
    yield
    
    # Shutdown: cleanup if needed
    print("Shutting down AI Spotlight Backend...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Spotlight Backend",
    description="AI-powered search with RAG and web search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    use_web_search: bool = False
    use_rag: bool = False
    max_tokens: Optional[int] = 512

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    search_results: List[Dict] = []

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_handler.model is not None,
        "rag_indexed": rag_engine.index is not None
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main endpoint for processing queries

    Args:
        request: QueryRequest with query text and options

    Returns:
        QueryResponse with AI-generated response and sources
    """
    try:
        context_parts = []
        sources = []
        search_results = []

        # Web search
        if request.use_web_search:
            print(f"Performing web search for: {request.query}")
            search_results = web_search.search_web(request.query)
            if search_results:
                context_parts.append(web_search.format_search_results(search_results))

        # RAG retrieval
        if request.use_rag:
            print(f"Retrieving RAG context for: {request.query}")
            rag_results = rag_engine.retrieve_context(request.query)
            if rag_results:
                rag_context = "Relevant Documents:\n\n"
                for i, result in enumerate(rag_results, 1):
                    rag_context += f"{i}. {result['text'][:500]}...\n"
                    rag_context += f"   (Source: {result['metadata']['file']}, "
                    rag_context += f"Chunk {result['metadata']['chunk_id'] + 1}/{result['metadata']['total_chunks']})\n\n"
                    sources.append(result['metadata'])
                context_parts.append(rag_context)

        # Construct prompt
        if context_parts:
            prompt = f"Context:\n\n{''.join(context_parts)}\n\nQuestion: {request.query}\n\nAnswer:"
        else:
            prompt = request.query

        # Generate response
        print("Generating response...")
        response = model_handler.generate_response(
            prompt,
            max_tokens=request.max_tokens
        )

        return QueryResponse(
            response=response,
            sources=sources,
            search_results=search_results
        )

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG document upload endpoint
@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for RAG indexing

    Args:
        file: Document file (.txt, .pdf, .md)

    Returns:
        Success message
    """
    try:
        # Check file extension
        allowed_extensions = ['.txt', '.pdf', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Save file
        file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File saved: {file_path}")

        # Re-index documents
        print("Re-indexing documents...")
        rag_engine.index_documents()

        return {
            "message": f"Document '{file.filename}' uploaded and indexed successfully",
            "total_chunks": len(rag_engine.documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Re-index endpoint
@app.post("/rag/reindex")
async def reindex_documents():
    """Re-index all documents in the documents directory"""
    try:
        rag_engine.index_documents()
        return {
            "message": "Documents re-indexed successfully",
            "total_chunks": len(rag_engine.documents),
            "total_documents": len(set(m['file'] for m in rag_engine.doc_metadata))
        }
    except Exception as e:
        print(f"Error re-indexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get indexed documents
@app.get("/rag/documents")
async def get_indexed_documents():
    """Get list of indexed documents"""
    if not rag_engine.doc_metadata:
        return {"documents": [], "total_chunks": 0}

    documents = {}
    for metadata in rag_engine.doc_metadata:
        file_name = metadata['file']
        if file_name not in documents:
            documents[file_name] = {
                'filename': file_name,
                'total_chunks': metadata['total_chunks']
            }

    return {
        "documents": list(documents.values()),
        "total_chunks": len(rag_engine.documents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
