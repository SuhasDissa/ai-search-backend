# Single-stage build for AI Spotlight Backend with GPU support
FROM huggingface/transformers-pytorch-gpu:latest

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install only the packages not already in base image
# Base image already includes: torch, transformers, tokenizers, datasets
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    accelerate \
    sentence-transformers \
    faiss-gpu-cu12 \
    langchain \
    requests \
    python-multipart \
    duckduckgo-search \
    pypdf \
    python-dotenv \
    aiohttp \
    beautifulsoup4 \
    scikit-learn \
    numpy

# Download models to cache
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    from sentence_transformers import SentenceTransformer; \
    print('Downloading Qwen tokenizer and model...'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    print('Downloading embedding model...'); \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    print('All models cached successfully!')"

# Create directories for data persistence
RUN mkdir -p /app/data/documents /app/data/faiss_index && \
    chmod -R 755 /app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set up model cache directory for appuser
RUN mkdir -p /home/appuser/.cache && \
    cp -r /root/.cache/* /home/appuser/.cache/ 2>/dev/null || true && \
    chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY app/ ./app/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DOCUMENTS_DIR=/app/data/documents \
    FAISS_INDEX_PATH=/app/data/faiss_index/index

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
