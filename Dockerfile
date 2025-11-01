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
    fastapi==0.120.4 \
    uvicorn[standard]==0.38.0 \
    accelerate==1.11.0 \
    sentence-transformers==5.1.2 \
    faiss-gpu==1.12.0 \
    langchain==1.0.3 \
    requests==2.32.5 \
    python-multipart==0.0.20 \
    duckduckgo-search==8.1.1 \
    pypdf==6.1.3 \
    python-dotenv==1.2.1 \
    aiohttp==3.11.11 \
    beautifulsoup4==4.12.3 \
    scikit-learn==1.6.1 \
    numpy==2.2.3

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
