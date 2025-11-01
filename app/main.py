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
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load models on startup
    logger.info("=" * 60)
    logger.info("Starting AI Spotlight Backend...")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info("=" * 60)

    model_handler.load_model()

    # Set model handler for web search query optimization
    web_search.set_model_handler(model_handler)
    logger.info("Web search query optimization enabled")

    # Try to load existing RAG index
    if os.path.exists(f"{settings.FAISS_INDEX_PATH}.index"):
        rag_engine.load_index()
    else:
        logger.info("No existing RAG index found. Upload documents to create one.")

    logger.info("AI Spotlight Backend startup complete")

    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down AI Spotlight Backend...")

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

class WebSearchMetadata(BaseModel):
    original_query: str
    optimized_query: Optional[str] = None
    urls_fetched: int = 0
    chunks_ranked: int = 0
    avg_relevance_score: Optional[float] = None
    cache_used: bool = False
    search_time_ms: Optional[int] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    search_results: List[Dict] = []
    web_search_metadata: Optional[WebSearchMetadata] = None
    processing_time_ms: Optional[int] = None

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
        import time
        request_start = time.time()

        context_parts = []
        sources = []
        search_results = []
        web_metadata = None

        logger.info(f"Query request: '{request.query}' (web_search={request.use_web_search}, rag={request.use_rag})")

        # Web search with enhanced context
        if request.use_web_search:
            logger.info("Performing enhanced web search")
            # Use enhanced_search which:
            # 1. Uses Qwen to optimize the query
            # 2. Fetches full webpage content
            # 3. Ranks content with TF-IDF
            # 4. Returns compact context + metadata
            compact_context, metadata = web_search.enhanced_search(request.query)
            if compact_context:
                context_parts.append("Web Search Results:\n\n" + compact_context)
                web_metadata = WebSearchMetadata(**metadata)
                logger.debug(f"Web search added {len(compact_context)} chars of context")

        # RAG retrieval
        if request.use_rag:
            logger.info("Retrieving RAG context")
            rag_results = rag_engine.retrieve_context(request.query)
            if rag_results:
                rag_context = "Relevant Documents:\n\n"
                for i, result in enumerate(rag_results, 1):
                    rag_context += f"{i}. {result['text'][:500]}...\n"
                    rag_context += f"   (Source: {result['metadata']['file']}, "
                    rag_context += f"Chunk {result['metadata']['chunk_id'] + 1}/{result['metadata']['total_chunks']})\n\n"
                    sources.append(result['metadata'])
                context_parts.append(rag_context)
                logger.debug(f"RAG added {len(rag_results)} sources")

        # Construct prompt
        if context_parts:
            prompt = f"Context:\n\n{''.join(context_parts)}\n\nQuestion: {request.query}\n\nAnswer:"
        else:
            prompt = request.query

        # Generate response
        logger.info("Generating response with LLM")
        response = model_handler.generate_response(
            prompt,
            max_tokens=request.max_tokens
        )

        processing_time = int((time.time() - request_start) * 1000)
        logger.info(f"Query completed in {processing_time}ms")

        return QueryResponse(
            response=response,
            sources=sources,
            search_results=search_results,
            web_search_metadata=web_metadata,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
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

        logger.info(f"File saved: {file_path}")

        # Re-index documents
        logger.info("Re-indexing documents...")
        rag_engine.index_documents()

        logger.info(f"Document '{file.filename}' uploaded and indexed successfully")
        return {
            "message": f"Document '{file.filename}' uploaded and indexed successfully",
            "total_chunks": len(rag_engine.documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Re-index endpoint
@app.post("/rag/reindex")
async def reindex_documents():
    """Re-index all documents in the documents directory"""
    try:
        logger.info("Manual re-index requested")
        rag_engine.index_documents()
        result = {
            "message": "Documents re-indexed successfully",
            "total_chunks": len(rag_engine.documents),
            "total_documents": len(set(m['file'] for m in rag_engine.doc_metadata))
        }
        logger.info(f"Re-index complete: {result['total_chunks']} chunks, {result['total_documents']} documents")
        return result
    except Exception as e:
        logger.error(f"Error re-indexing documents: {e}", exc_info=True)
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
