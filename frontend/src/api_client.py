from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator, List, Callable, Union
import os
import json
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClientInterface(ABC):
    """Interface defining the contract for API clients."""
    
    @abstractmethod
    def get_response(self, user_input: str) -> str:
        """Get a response from the API for the given user input."""
        pass
    
    @abstractmethod
    async def get_streaming_response(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """Get a streaming response from the API for the given user input.
        
        Args:
            user_input: The user's input message
            callback: Optional callback function to handle each chunk of the response
            
        Returns:
            AsyncGenerator[str, None]: An async generator yielding response chunks
        """
        pass

class RAGAPIClient(APIClientInterface):
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the API client.
        
        Args:
            base_url: Optional base URL for the API. If not provided, will use API_BASE_URL from .env
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        load_dotenv()
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def get_response(self, user_input: str) -> str:
        """Get a response from the RAG API.
        
        Args:
            user_input: The user's input message
            
        Returns:
            str: The API response
        """
        # Create a synchronous wrapper around the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_response_async(user_input))
        finally:
            loop.close()
    
    async def get_streaming_response(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """Get a streaming response from the API for the given user input.
        
        Args:
            user_input: The user's input message
            callback: Optional callback function to handle each chunk of the response
            
        Returns:
            AsyncGenerator[str, None]: An async generator yielding response chunks
        """
        async for chunk in self._get_streaming_response_async(user_input, callback):
            yield chunk
    
    async def _get_response_async(self, user_input: str) -> str:
        """Asynchronously get a response from the RAG API.
        
        Args:
            user_input: The user's input message
            
        Returns:
            str: The API response
        """
        url = f"{self.base_url}/chat"
        payload = {
            "user_prompt": user_input
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("response", "")
                        else:
                            error_text = await response.text()
                            logger.error(f"API error (attempt {attempt+1}/{self.max_retries}): {response.status} - {error_text}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                            else:
                                raise Exception(f"API error: {response.status} - {error_text}")
            except Exception as e:
                logger.error(f"Request error (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Failed to get response after {self.max_retries} attempts: {str(e)}")
    
    async def _get_streaming_response_async(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """Asynchronously get a streaming response from the RAG API.
        
        Args:
            user_input: The user's input message
            callback: Optional callback function to handle each chunk of the response
            
        Returns:
            AsyncGenerator[str, None]: An async generator yielding response chunks
        """
        url = f"{self.base_url}/chat/stream"
        payload = {
            "user_prompt": user_input
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: '):
                                    data = line[6:]  # Remove 'data: ' prefix
                                    if data == '[DONE]':
                                        break
                                    try:
                                        json_data = json.loads(data)
                                        if 'content' in json_data:
                                            content = json_data['content']
                                            if callback:
                                                callback(content)
                                            yield content
                                        elif 'error' in json_data:
                                            error_msg = json_data['error']
                                            logger.error(f"Streaming error: {error_msg}")
                                            yield f"Error: {error_msg}"
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse JSON: {data}")
                            return
                        else:
                            error_text = await response.text()
                            logger.error(f"API error (attempt {attempt+1}/{self.max_retries}): {response.status} - {error_text}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                            else:
                                raise Exception(f"API error: {response.status} - {error_text}")
            except Exception as e:
                logger.error(f"Request error (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Failed to get streaming response after {self.max_retries} attempts: {str(e)}")

class APIClientFactory:
    """Factory for creating API client instances."""
    
    @staticmethod
    def create_client(client_type: str = "rag", **kwargs) -> APIClientInterface:
        """Create an API client instance.
        
        Args:
            client_type: Type of client to create ("rag" by default)
            **kwargs: Additional arguments to pass to the client constructor
            
        Returns:
            APIClientInterface: An instance of the requested client type
            
        Raises:
            ValueError: If the requested client type is not supported
        """
        if client_type.lower() == "rag":
            return RAGAPIClient(**kwargs)
        raise ValueError(f"Unsupported client type: {client_type}")

# Global function for backward compatibility
def get_response(user_input: str) -> str:
    """Get a response from the API.
    
    This is a convenience function that uses the default RAG client.
    
    Args:
        user_input: The user's input message
        
    Returns:
        str: The API response
    """
    client = APIClientFactory.create_client()
    return client.get_response(user_input)

# New global function for streaming responses
async def get_streaming_response(user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
    """Get a streaming response from the API.
    
    This is a convenience function that uses the default RAG client.
    
    Args:
        user_input: The user's input message
        callback: Optional callback function to handle each chunk of the response
        
    Returns:
        AsyncGenerator[str, None]: An async generator yielding response chunks
    """
    client = APIClientFactory.create_client()
    async for chunk in client.get_streaming_response(user_input, callback):
        yield chunk
