import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import numpy as np
from pipeline import (
    Config,
    ConfigError,
    MilvusError,
    ModelError,
    MilvusConnector,
    MarkdownProcessor,
    load_config,
    process_documents
)

# Test data
SAMPLE_CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'keyword_model': 'all-MiniLM-L6-v2',
    'chunk_size': 500,
    'chunk_overlap': 100,
    'markdown_folder': '../docs',
    'milvus': {
        'host': 'localhost',
        'port': 19530
    },
    'collection': {
        'name': 'test_collection',
        'schema': {
            'fields': [
                {
                    'name': 'id',
                    'data_type': 'INT64',
                    'description': 'Primary key',
                    'is_primary': True,
                    'auto_id': True
                },
                {
                    'name': 'embedding',
                    'data_type': 'FLOAT_VECTOR',
                    'description': 'Document embedding',
                    'dim': 384
                },
                {
                    'name': 'content',
                    'data_type': 'VARCHAR',
                    'description': 'Document content',
                    'max_length': 65535
                },
                {
                    'name': 'metadata',
                    'data_type': 'JSON',
                    'description': 'Document metadata'
                },
                {
                    'name': 'keywords',
                    'data_type': 'ARRAY',
                    'description': 'Document keywords',
                    'element_type': 'VARCHAR'
                },
                {
                    'name': 'created_at',
                    'data_type': 'VARCHAR',
                    'description': 'Creation timestamp'
                }
            ]
        }
    }
}

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config.from_dict(SAMPLE_CONFIG)

@pytest.fixture
def mock_milvus():
    """Mock Milvus connections and collections."""
    with patch('pipeline.connections') as mock_conn, \
         patch('pipeline.Collection') as mock_collection, \
         patch('pipeline.utility') as mock_utility:
        mock_utility.has_collection.return_value = False
        mock_collection.return_value = MagicMock()
        yield mock_conn, mock_collection, mock_utility

@pytest.fixture
def mock_models():
    """Mock sentence transformers and keybert models."""
    with patch('pipeline.SentenceTransformer') as mock_st, \
         patch('pipeline.KeyBERT') as mock_kb:
        # Create a mock for the SentenceTransformer instance
        mock_st_instance = MagicMock()
        # Configure encode to return a numpy array with shape (2, 384) for two chunks
        mock_st_instance.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        mock_st.return_value = mock_st_instance
        
        # Create a mock for the KeyBERT instance
        mock_kb_instance = MagicMock()
        mock_kb_instance.extract_keywords.return_value = [('test', 0.5)]
        mock_kb.return_value = mock_kb_instance
        
        yield mock_st, mock_kb

def test_config_validation():
    """Test configuration validation."""
    # Test missing required field
    invalid_config = SAMPLE_CONFIG.copy()
    del invalid_config['embedding_model']
    with pytest.raises(ConfigError, match="Missing required field"):
        Config.from_dict(invalid_config)

    # Test invalid chunk size
    invalid_config = SAMPLE_CONFIG.copy()
    invalid_config['chunk_size'] = -1
    with pytest.raises(ConfigError, match="chunk_size must be a positive integer"):
        Config.from_dict(invalid_config)

    # Test invalid chunk overlap
    invalid_config = SAMPLE_CONFIG.copy()
    invalid_config['chunk_overlap'] = -1
    with pytest.raises(ConfigError, match="chunk_overlap must be a non-negative integer"):
        Config.from_dict(invalid_config)

def test_milvus_connector_initialization(config, mock_milvus, mock_models):
    """Test MilvusConnector initialization."""
    connector = MilvusConnector(config)
    assert connector.config == config
    mock_milvus[0].connect.assert_called_once_with(
        "default", host=config.milvus_host, port=config.milvus_port
    )

def test_milvus_connector_collection_creation(config, mock_milvus):
    """Test collection creation in MilvusConnector."""
    connector = MilvusConnector(config)
    connector.ensure_collection_exists()
    mock_milvus[1].assert_called_once_with(
        name=config.collection_name,
        schema=connector._create_collection_schema()
    )

def test_milvus_connector_data_insertion(config, mock_milvus, mock_models):
    """Test data insertion in MilvusConnector."""
    connector = MilvusConnector(config)
    chunks = ["Test chunk 1", "Test chunk 2"]
    metadata_list = [{"source": "test1"}, {"source": "test2"}]
    
    connector.insert_data(chunks, metadata_list)
    
    # Verify model calls
    mock_models[0].return_value.encode.assert_called_once_with(chunks)
    assert mock_models[1].return_value.extract_keywords.call_count == len(chunks)
    
    # Verify collection operations
    collection = mock_milvus[1].return_value
    collection.insert.assert_called_once()
    collection.flush.assert_called_once()

def test_markdown_processor():
    """Test MarkdownProcessor functionality."""
    # Create test markdown file
    test_content = """---
title: Test Document
author: Test Author
---

# Test Content
This is a test document.
"""
    test_file = Path("test_docs/test.md")
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(test_content)

    try:
        # Test parsing
        metadata, content = MarkdownProcessor.parse_markdown(str(test_file))
        assert metadata['title'] == 'Test Document'
        assert metadata['author'] == 'Test Author'
        assert '<h1>Test Content</h1>' in content

        # Test chunking
        chunks = MarkdownProcessor.chunk_text(content, chunk_size=100, overlap=20)
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)
    finally:
        # Cleanup
        test_file.unlink()
        test_file.parent.rmdir()

def test_load_config(tmp_path):
    """Test configuration loading."""
    # Create test config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(SAMPLE_CONFIG))

    # Test loading
    config = load_config(str(config_file))
    assert config.embedding_model == SAMPLE_CONFIG['embedding_model']
    assert config.chunk_size == SAMPLE_CONFIG['chunk_size']

def test_process_documents(config, mock_milvus, mock_models, tmp_path):
    """Test document processing pipeline."""
    # Create test markdown files
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()
    
    test_files = [
        ("doc1.md", "Test document 1"),
        ("doc2.md", "Test document 2")
    ]
    
    for filename, content in test_files:
        file_path = docs_dir / filename
        file_path.write_text(f"---\ntitle: {filename}\n---\n\n{content}")

    # Update config with test directory
    config.markdown_folder = str(docs_dir)
    
    # Create connector and process documents
    connector = MilvusConnector(config)
    process_documents(config, connector)
    
    # Verify collection operations
    collection = mock_milvus[1].return_value
    # Each document is processed and inserted separately
    assert collection.insert.call_count == len(test_files)
    # Flush is called after each document insertion
    assert collection.flush.call_count == len(test_files)

def test_error_handling():
    """Test error handling in various components."""
    # Test ConfigError
    with pytest.raises(ConfigError):
        Config.from_dict({})

    # Test MilvusError
    with patch('pipeline.connections') as mock_conn:
        mock_conn.connect.side_effect = Exception("Connection failed")
        with pytest.raises(ModelError):
            MilvusConnector(Config.from_dict(SAMPLE_CONFIG))

    # Test ModelError
    with patch('pipeline.SentenceTransformer') as mock_st:
        mock_st.side_effect = Exception("Model loading failed")
        with pytest.raises(ModelError):
            MilvusConnector(Config.from_dict(SAMPLE_CONFIG)) 