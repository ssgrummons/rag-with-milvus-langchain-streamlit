# RAG Pipeline

This pipeline processes markdown documents and uploads them to Milvus for use in a RAG (Retrieval-Augmented Generation) application. It handles document chunking, embedding generation, and keyword extraction.

## Features

- Processes markdown files with frontmatter
- Generates embeddings using sentence-transformers
- Extracts keywords using KeyBERT
- Chunks documents using LangChain's RecursiveCharacterTextSplitter
- Uploads processed data to Milvus with proper schema validation
- Comprehensive error handling and logging
- Configuration validation

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- Docker and Docker Compose (for running services)
- Access to the deployment services (Milvus, etc.)

## Installation

1. Clone the repository and navigate to the pipelines directory:
```bash
cd pipelines
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Configuration

The pipeline is configured using a `config.yaml` file. Here's an example configuration:

```yaml
embedding_model: "all-MiniLM-L6-v2"
keyword_model: "all-MiniLM-L6-v2"
chunk_size: 500
chunk_overlap: 100
markdown_folder: "../docs"
milvus:
  host: "localhost"
  port: "19530"
collection: 
  name: "knowledge_base"
  schema:
    fields:
      - name: "id"
        data_type: "INT64"
        description: "Unique identifier for the document"
        is_primary: true
        auto_id: true
        required: true
      - name: "embedding"
        data_type: "FLOAT_VECTOR"
        description: "Embedding of the chunk"
        dim: 384
        required: true
      - name: "content"
        data_type: "String"
        description: "Plain text content of the chunk"
        required: true
      - name: "metadata"
        data_type: "JSON"
        description: "Metadata of the document"
        required: false
      - name: "keywords"
        data_type: "Array"
        description: "Keywords of the chunk"
        required: false
      - name: "created_at"
        data_type: "String"
        description: "Created at date of the record"
        required: true
```

## Running the Pipeline

### Manual Execution

1. Ensure the deployment services are running:
```bash
cd ../deployment
docker-compose up -d
```

2. Activate the Poetry environment:
```bash
cd ../pipelines
poetry shell
```

3. Run the pipeline:
```bash
python pipeline.py
```

### Running as a Cron Job

1. Create a shell script (e.g., `run_pipeline.sh`):
```bash
#!/bin/bash
cd /path/to/pipelines
poetry run python pipeline.py >> /path/to/pipeline.log 2>&1
```

2. Make the script executable:
```bash
chmod +x run_pipeline.sh
```

3. Add a cron job (e.g., to run daily at 2 AM):
```bash
0 2 * * * /path/to/run_pipeline.sh
```

## Error Handling

The pipeline includes comprehensive error handling for common scenarios:
- Configuration validation errors
- Model loading failures
- Milvus connection issues
- File processing errors
- Data insertion failures

All errors are logged with appropriate context and severity levels.

## Development

### Running Tests

The project includes a comprehensive test suite. To run the tests:

```bash
# Run all tests
poetry run pytest

# Run tests with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_pipeline.py

# Run tests with coverage report
poetry run pytest --cov=pipeline tests/
```

### Test Structure

- `tests/test_pipeline.py`: Contains all unit tests for the pipeline components
- Tests cover:
  - Configuration validation
  - Milvus connector functionality
  - Markdown processing
  - Error handling
  - Data insertion
  - Document processing pipeline

### Test Dependencies

The following test dependencies are included:
- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-mock: Mocking utilities

### Writing Tests

When adding new tests:
1. Place test files in the `tests/` directory
2. Use pytest fixtures for common setup
3. Mock external dependencies (Milvus, models)
4. Include both positive and negative test cases
5. Clean up test resources after each test

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here] 