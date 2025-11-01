# AI Spotlight Backend

AI-powered search backend with RAG (Retrieval-Augmented Generation) and web search capabilities using Qwen 2 language model.

## Features

- **AI-Powered Responses**: Uses Qwen2.5-1.5B-Instruct model for intelligent responses
- **RAG Support**: Index and search through your documents (.txt, .pdf, .md)
- **Web Search**: Integrated DuckDuckGo search for up-to-date information
- **REST API**: FastAPI-based service for easy integration
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

```
backend-ai-search/
├── docker-compose.yml    # Docker orchestration
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
├── app/
│   ├── main.py         # FastAPI application
│   ├── model_handler.py # Qwen model management
│   ├── rag_engine.py   # RAG implementation
│   ├── web_search.py   # Web search integration
│   └── config.py       # Configuration settings
└── data/
    └── documents/      # Document storage for RAG
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM (8GB+ recommended)
- CPU with AVX support (or GPU for better performance)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend-ai-search
```

2. Build and start the service:
```bash
docker-compose build
docker-compose up -d
```

3. Check the logs:
```bash
docker-compose logs -f
```

The service will be available at `http://localhost:8000`

### First Run

The first time you start the service, it will download the Qwen model (~1GB). This may take several minutes depending on your internet connection.

## API Endpoints

### Health Check
```bash
GET /health
```

Returns the service status and model information.

### Query Endpoint
```bash
POST /query
Content-Type: application/json

{
  "query": "What is quantum computing?",
  "use_web_search": true,
  "use_rag": false,
  "max_tokens": 512
}
```

Response:
```json
{
  "response": "Quantum computing is...",
  "sources": [],
  "search_results": [...]
}
```

### Upload Document
```bash
POST /rag/upload
Content-Type: multipart/form-data

file: document.pdf
```

Upload a document for RAG indexing.

### Re-index Documents
```bash
POST /rag/reindex
```

Re-index all documents in the `data/documents/` directory.

### Get Indexed Documents
```bash
GET /rag/documents
```

List all indexed documents and their chunk counts.

## Configuration

Edit `docker-compose.yml` to customize:

- `MODEL_NAME`: Change the Qwen model variant
- `DEVICE`: Set to `cuda` for GPU acceleration
- Port mapping: Change `8000:8000` to use a different port

For advanced configuration, edit `app/config.py`:

```python
# Model settings
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"
MAX_LENGTH = 512

# RAG settings
CHUNK_SIZE = 500
TOP_K = 3

# Web search settings
MAX_SEARCH_RESULTS = 5
```

## GPU Support

To enable GPU acceleration:

1. Install NVIDIA Docker runtime
2. Uncomment the GPU section in `docker-compose.yml`
3. Set `DEVICE=cuda` in environment variables
4. Rebuild and restart: `docker-compose up -d --build`

## RAG Usage

### Adding Documents

1. Place documents in `data/documents/` directory:
```bash
cp my-document.pdf data/documents/
```

2. Re-index:
```bash
curl -X POST http://localhost:8000/rag/reindex
```

Or upload via API:
```bash
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@my-document.pdf"
```

### Supported Formats

- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.md` - Markdown files

## Testing

Test the API with curl:

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain neural networks", "use_web_search": false, "use_rag": false}'

# Query with web search
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest AI news", "use_web_search": true, "use_rag": false}'

# Query with RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about...?", "use_web_search": false, "use_rag": true}'
```

## Troubleshooting

### Model Loading Issues
- Ensure you have sufficient RAM (4GB minimum)
- Check Docker logs: `docker-compose logs -f`
- Try using a smaller model variant

### Slow Performance
- Use GPU if available (see GPU Support section)
- Reduce `MAX_LENGTH` in config
- Use quantized models (requires code modifications)

### Connection Refused
- Check if the container is running: `docker-compose ps`
- Verify port 8000 is not in use: `netstat -tuln | grep 8000`
- Check firewall settings

### Out of Memory
- Increase Docker memory limit
- Use smaller model (Qwen2-1.5B instead of 2B)
- Reduce chunk size for RAG

## Development

To run without Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
cd app
python -m uvicorn main:app --reload
```

## Performance Optimization

- **Quantization**: Use 4-bit or 8-bit quantized models
- **Caching**: Enable response caching for common queries
- **Batch Processing**: Process multiple queries together
- **vLLM**: Consider using vLLM for better inference performance

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: [Read the docs]
