# Stage 1: Builder
# Use the CUDA 12.1 development image to build our venv
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install Python 3.11 and venv
# Use --no-install-recommends to keep it slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip

# Set dedicated cache directory for models
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Install dependencies from requirements.txt into the venv
# This now MUST include torch, transformers, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache models
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    from sentence_transformers import SentenceTransformer; \
    print('Downloading Qwen tokenizer and model...'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    print('Downloading embedding model...'); \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    print('All models cached successfully!')"

# ---

# Stage 2: Final Image
# Use the smaller CUDA 12.1 runtime image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create and set permissions for persistent data directories
RUN mkdir -p /app/data/documents /app/data/faiss_index && \
    chown -R appuser:appuser /app/data

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the clean model cache from the builder stage
COPY --from=builder /app/model_cache /home/appuser/.cache
RUN chown -R appuser:appuser /home/appuser/.cache

# Copy application code and set ownership
COPY --chown=appuser:appuser app/ ./app/

# Switch to the non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables for the running container
ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/home/appuser/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DOCUMENTS_DIR=/app/data/documents
ENV FAISS_INDEX_PATH=/app/data/faiss_index/index

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
