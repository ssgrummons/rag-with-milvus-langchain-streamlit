import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
import os
import sys

# Add the parent directory to Python path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import ChatApp

@pytest.fixture
def mock_streamlit():
    """Fixture to mock Streamlit functionality."""
    with patch('streamlit.set_page_config') as mock_set_page_config, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.chat_input') as mock_chat_input, \
         patch('streamlit.chat_message') as mock_chat_message, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.session_state', new_callable=dict) as mock_session_state:
        
        # Setup mock chat_message context manager
        mock_chat_message.return_value.__enter__.return_value = MagicMock()
        mock_chat_message.return_value.__exit__.return_value = None
        
        yield {
            'set_page_config': mock_set_page_config,
            'title': mock_title,
            'chat_input': mock_chat_input,
            'chat_message': mock_chat_message,
            'markdown': mock_markdown,
            'session_state': mock_session_state
        }

@pytest.fixture
def mock_api_client():
    """Fixture to mock API client."""
    with patch('src.app.get_response') as mock_get_response:
        mock_get_response.return_value = "Test response"
        yield mock_get_response

@pytest.fixture
def chat_app():
    """Fixture to create a ChatApp instance."""
    with patch('src.app.load_dotenv'), \
         patch('src.app.os.getenv', return_value='8501'):
        return ChatApp()

def test_chat_app_initialization(chat_app, mock_streamlit):
    """Test ChatApp initialization."""
    assert chat_app.port == 8501
    mock_streamlit['set_page_config'].assert_called_once_with(
        page_title="Chat with RAG",
        layout="wide"
    )

def test_initialize_session_state(chat_app, mock_streamlit):
    """Test session state initialization."""
    chat_app._initialize_session_state()
    assert 'messages' in mock_streamlit['session_state']
    assert mock_streamlit['session_state']['messages'] == []

def test_handle_user_input(chat_app, mock_streamlit, mock_api_client):
    """Test handling of user input."""
    user_input = "Hello, how are you?"
    chat_app._handle_user_input(user_input)
    
    # Verify user message was added
    assert len(mock_streamlit['session_state']['messages']) == 2
    assert mock_streamlit['session_state']['messages'][0]['role'] == 'user'
    assert mock_streamlit['session_state']['messages'][0]['content'] == user_input
    
    # Verify assistant response was added
    assert mock_streamlit['session_state']['messages'][1]['role'] == 'assistant'
    assert mock_streamlit['session_state']['messages'][1]['content'] == "Test response"
    
    # Verify markdown was called for both messages
    assert mock_streamlit['markdown'].call_count == 2

def test_display_chat_history(chat_app, mock_streamlit):
    """Test displaying chat history."""
    # Setup test messages
    mock_streamlit['session_state']['messages'] = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there'}
    ]
    
    chat_app._display_chat_history()
    
    # Verify chat_message was called for each message
    assert mock_streamlit['chat_message'].call_count == 2
    assert mock_streamlit['markdown'].call_count == 2

def test_run_with_user_input(chat_app, mock_streamlit, mock_api_client):
    """Test the main run method with user input."""
    mock_streamlit['chat_input'].return_value = "Hello"
    
    chat_app.run()
    
    # Verify title was set
    mock_streamlit['title'].assert_called_once_with("Chat with the RAG Model")
    
    # Verify messages were handled
    assert len(mock_streamlit['session_state']['messages']) == 2

def test_run_without_user_input(chat_app, mock_streamlit):
    """Test the main run method without user input."""
    mock_streamlit['chat_input'].return_value = None
    
    chat_app.run()
    
    # Verify title was set
    mock_streamlit['title'].assert_called_once_with("Chat with the RAG Model")
    
    # Verify no messages were added
    assert len(mock_streamlit['session_state']['messages']) == 0 