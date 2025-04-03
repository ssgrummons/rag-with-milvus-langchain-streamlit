import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import streamlit as st
import os
import sys
import asyncio

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
         patch('streamlit.empty') as mock_empty, \
         patch('streamlit.sidebar.checkbox') as mock_checkbox, \
         patch('streamlit.session_state', new_callable=dict) as mock_session_state:
        
        # Setup mock chat_message context manager
        mock_chat_message.return_value.__enter__.return_value = MagicMock()
        mock_chat_message.return_value.__exit__.return_value = None
        
        # Setup mock empty context manager
        mock_empty.return_value = MagicMock()
        
        # Setup mock checkbox to return True for streaming
        mock_checkbox.return_value = True
        
        yield {
            'set_page_config': mock_set_page_config,
            'title': mock_title,
            'chat_input': mock_chat_input,
            'chat_message': mock_chat_message,
            'markdown': mock_markdown,
            'empty': mock_empty,
            'checkbox': mock_checkbox,
            'session_state': mock_session_state
        }

@pytest.fixture
def mock_api_client():
    """Fixture to mock API client."""
    # Mock the get_response function
    with patch('src.app.get_response') as mock_get_response, \
         patch('src.app.get_streaming_response') as mock_get_streaming_response:
        
        # Setup the regular response mock
        mock_get_response.return_value = "Test response"
        
        # Setup the streaming response mock
        async def mock_streaming_generator():
            yield "Hello"
            yield " world"
            yield "!"
        
        # Create an AsyncMock that returns our generator
        mock_get_streaming_response.return_value = mock_streaming_generator()
        
        yield {
            'get_response': mock_get_response,
            'get_streaming_response': mock_get_streaming_response
        }

@pytest.fixture
def chat_app(mock_streamlit):
    """Fixture to create a ChatApp instance."""
    with patch('src.app.load_dotenv'), \
         patch('src.app.os.getenv', return_value='8501'):
        app = ChatApp(use_streaming=True)
        return app

def test_chat_app_initialization(chat_app, mock_streamlit):
    """Test ChatApp initialization."""
    # Verify port is set correctly
    assert chat_app.port == 8501
    
    # Verify page config was set
    mock_streamlit['set_page_config'].assert_called_once_with(
        page_title="Chat with RAG",
        layout="wide"
    )
    
    # Verify session state was initialized
    assert 'messages' in mock_streamlit['session_state']
    assert mock_streamlit['session_state']['messages'] == []

def test_initialize_session_state(chat_app, mock_streamlit):
    """Test session state initialization."""
    chat_app._initialize_session_state()
    assert 'messages' in mock_streamlit['session_state']
    assert mock_streamlit['session_state']['messages'] == []

def test_handle_user_input(chat_app, mock_streamlit, mock_api_client):
    """Test handling of user input."""
    user_input = "Hello, how are you?"
    
    # Create a mock placeholder
    mock_placeholder = MagicMock()
    mock_streamlit['empty'].return_value = mock_placeholder
    
    # Mock the _handle_streaming_response method directly
    with patch.object(chat_app, '_handle_streaming_response') as mock_handle_streaming:
        # Setup the mock to update the session state
        def mock_handle_streaming_side_effect(user_input, placeholder):
            # Simulate the streaming response
            full_response = "Hello world!"
            placeholder.markdown(full_response)
            mock_streamlit['session_state']['messages'].append(
                {"role": "assistant", "content": full_response}
            )
        
        mock_handle_streaming.side_effect = mock_handle_streaming_side_effect
        
        # Call the method
        chat_app._handle_user_input(user_input)
    
    # Verify markdown was called for both messages
    assert mock_placeholder.markdown.call_count == 1  # Assistant response
    assert mock_streamlit['markdown'].call_count == 1  # User input

    # Verify the first call was with the user input
    mock_streamlit['markdown'].assert_called_once_with(user_input)

    # Verify the second call was with the assistant response
    mock_placeholder.markdown.assert_called_once_with("Hello world!")


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