# RAG Backend Service

This is the backend service for the RAG (Retrieval-Augmented Generation) application. It provides a FastAPI-based API that handles document retrieval and chat interactions.

## Features

- FastAPI-based REST API
- Integration with Milvus vector database
- Integration with Ollama for LLM inference
- Streaming chat responses
- Health check endpoint
- Configuration management
- CORS support

## Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- Docker (optional, for containerized deployment)
- Milvus vector database
- Ollama with qwen2:7b  model

## Local Development Setup

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository and navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Create a `.env` file in the `src` directory with the following content:
   ```env
   # Milvus Configuration
   MILVUS_HOST=localhost
   MILVUS_PORT=19530

   # Ollama Configuration
   OLLAMA_HOST=http://host.docker.internal:11434
   OLLAMA_MODEL=qwen2:7b 

   # RAG Configuration
   CHUNK_SIZE=500
   CHUNK_OVERLAP=100
   MAX_TOKENS=2048
   TEMPERATURE=0.7

   # Logging Configuration
   LOG_LEVEL=INFO

   # Security Configuration
   CORS_ORIGINS=http://localhost:8501,http://localhost:3000
   ```

5. Run the application locally:
   ```bash
   poetry run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`.

## Docker Deployment

### Option 1: Using Docker Compose (Recommended)

1. Create a `.env` file in the `src` directory as described above.

2. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

### Option 2: Manual Docker Build

1. Build the Docker image:
   ```bash
   docker build -t rag-backend .
   ```

2. Run the container with environment variables:
   ```bash
   docker run -d \
     --name rag-backend \
     -p 8000:8000 \
     --env-file src/.env \
     rag-backend
   ```

### Option 3: Using Docker with Environment Variables

If you prefer to pass environment variables directly:

```bash
docker run -d \
  --name rag-backend \
  -p 8000:8000 \
  -e MILVUS_HOST=localhost \
  -e MILVUS_PORT=19530 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen2:7b  \
  -e CHUNK_SIZE=500 \
  -e CHUNK_OVERLAP=100 \
  -e MAX_TOKENS=2048 \
  -e TEMPERATURE=0.7 \
  -e LOG_LEVEL=INFO \
  -e CORS_ORIGINS=http://localhost:8501,http://localhost:3000 \
  rag-backend
```

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `GET /config`: Get current configuration
- `POST /chat`: Chat endpoint (non-streaming)
- `POST /chat/stream`: Streaming chat endpoint

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| MILVUS_HOST | Milvus service hostname | - | Yes |
| MILVUS_PORT | Milvus service port | - | Yes |
| OLLAMA_HOST | Ollama service URL | - | Yes |
| OLLAMA_MODEL | Default Ollama model | qwen2:7b  | No |
| CHUNK_SIZE | Size of text chunks | 500 | No |
| CHUNK_OVERLAP | Overlap between chunks | 100 | No |
| MAX_TOKENS | Maximum tokens for output | 2048 | No |
| TEMPERATURE | Model temperature | 0.7 | No |
| LOG_LEVEL | Logging level | INFO | No |
| CORS_ORIGINS | Allowed CORS origins | http://localhost:8501,http://localhost:3000 | No |

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Style

This project uses black for code formatting:

```bash
poetry run black .
```

## Troubleshooting

1. **Health Check Fails**
   - Ensure Milvus is running and accessible
   - Check Ollama service is running
   - Verify environment variables are set correctly

2. **CORS Issues**
   - Verify the frontend origin is included in CORS_ORIGINS
   - Check if the frontend is making requests to the correct backend URL

3. **Docker Issues**
   - Ensure Docker daemon is running
   - Check if ports are not already in use
   - Verify environment variables are properly set

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
