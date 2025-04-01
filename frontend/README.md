# RAG Chat Frontend

This is the frontend application for the RAG (Retrieval-Augmented Generation) chat system. It provides a Streamlit-based interface for interacting with the RAG model.

## Prerequisites

- Python 3.12
- Poetry (Python package manager)

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and navigate to the frontend directory:
```bash
cd frontend
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

Create a `.env` file in the `frontend` directory with the following variables:

```env
STREAMLIT_PORT=8501  # Port number for the Streamlit server
API_BASE_URL=http://localhost:8000  # URL of the backend API server
```

## Running the Application

To run the Streamlit application:

```bash
poetry run streamlit run src/app.py
```

The application will be available at `http://localhost:8501` by default.

## Testing

### Running Tests

To run the test suite:

```bash
poetry run pytest tests/ -v
```

For test coverage report:
```bash
poetry run pytest tests/ --cov=src --cov-report=term-missing
```

### Test Structure

The test suite includes:
- Unit tests for the ChatApp class
- Mocked Streamlit components
- API client integration tests
- Session state management tests

### Test Dependencies

The project uses Poetry for dependency management. Test dependencies are managed through the `pyproject.toml` file:

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-mock = "^3.11.0"
pytest-cov = "^4.1.0"  # Optional, for coverage reporting
```

## Project Structure

```
frontend/
├── src/
│   ├── app.py           # Main Streamlit application
│   └── api_client.py    # API client for backend communication
├── tests/
│   └── test_app.py      # Test suite for app.py
├── pyproject.toml       # Poetry project configuration
└── .env                # Environment variables
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
poetry run flake8 src/ tests/
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change the `STREAMLIT_PORT` in `.env` file
   - Or kill the existing process using the port

2. **API Connection Issues**
   - Verify the backend server is running
   - Check the `API_BASE_URL` in `.env` is correct

3. **Test Failures**
   - Ensure Poetry environment is activated
   - Run `poetry install` to update dependencies
   - Verify the test environment variables are set correctly

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
