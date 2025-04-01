# RAG Chat Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) chat system. It provides a FastAPI-based REST API for handling document processing, vector search, and chat interactions.

## Prerequisites

- Python 3.12
- Poetry (Python package manager)
- Docker and Docker Compose (for running services)

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and navigate to the backend directory:
```bash
cd backend
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Activate the Poetry shell:
```bash
poetry shell
```

## Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
MILVUS_HOST=localhost  # Milvus service hostname
MILVUS_PORT=19530     # Milvus service port
OLLAMA_HOST=http://localhost:11434  # Ollama service URL
```

## Running the Application

### Local Development

To run the FastAPI application locally:

```bash
poetry run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` by default.

### Docker

#### Building the Docker Image

To build the Docker image:

```bash
docker build -t rag-chat-backend .
```

#### Running with Docker

To run the application using Docker:

```bash
docker run -p 8000:8000 \
  -e MILVUS_HOST=milvus-standalone \
  -e MILVUS_PORT=19530 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  rag-chat-backend
```

Note: 
- The `-p 8000:8000` flag maps the container's port 8000 to your host machine's port 8000
- `host.docker.internal` is used to access the host machine from within the container
- Adjust the environment variables according to your service locations

#### Docker Compose

The backend service is part of the main docker-compose configuration in the deployment directory. To run it:

```bash
cd ../deployment
docker-compose up -d backend
```

### Docker Development

For development, you might want to mount the source code as a volume to enable hot-reloading:

```bash
docker run -p 8000:8000 \
  -e MILVUS_HOST=milvus-standalone \
  -e MILVUS_PORT=19530 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v $(pwd)/src:/app/src \
  rag-chat-backend
```

## API Documentation

Once the application is running, you can access:
- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## Testing

### Setting Up Tests

The project is set up with pytest for testing, but tests have not been developed yet. To create tests:

1. Create a `tests` directory:
```bash
mkdir tests
```

2. Create test files following the naming convention `test_*.py`:
```bash
touch tests/test_app.py
```

3. Example test structure:
```python
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

### Running Tests

To run the test suite:

```bash
poetry run pytest tests/ -v
```

For test coverage report:
```bash
poetry run pytest tests/ --cov=src --cov-report=term-missing
```

### Test Dependencies

The project uses Poetry for dependency management. Test dependencies are managed through the `pyproject.toml` file:

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.5"
pytest-cov = "^4.1.0"
httpx = "^0.26.0"  # For TestClient
```

## Project Structure

```
backend/
├── src/
│   ├── __init__.py     # Package initialization
│   ├── app.py          # Main FastAPI application
│   └── api/            # API routes and handlers
├── tests/              # Test suite (to be developed)
├── pyproject.toml      # Poetry project configuration
└── .env               # Environment variables
```

## Development

### Adding New Features

1. Create a new branch for your feature
2. Add tests for new functionality
3. Implement the feature
4. Run tests to ensure everything passes
5. Submit a pull request

### Code Style

The project follows PEP 8 guidelines. To check code style:

```bash
poetry run black src/ tests/
poetry run isort src/ tests/
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change the port in the run command
   - Or kill the existing process using the port

2. **Milvus Connection Issues**
   - Verify Milvus is running: `docker-compose ps`
   - Check Milvus logs: `docker-compose logs milvus-standalone`
   - Verify connection settings in `.env`

3. **Docker Build Issues**
   - Clean Docker cache: `docker builder prune`
   - Rebuild without cache: `docker build --no-cache -t rag-chat-backend .`

4. **Poetry Installation Issues**
   - Make sure Python 3.12 is installed and available
   - Try reinstalling Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
   - Clear Poetry cache: `poetry cache clear . --all`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]
