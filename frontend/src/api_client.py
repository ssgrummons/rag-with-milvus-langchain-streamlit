from abc import ABC, abstractmethod
from typing import Optional
import os
from dotenv import load_dotenv

class APIClientInterface(ABC):
    """Interface defining the contract for API clients."""
    
    @abstractmethod
    def get_response(self, user_input: str) -> str:
        """Get a response from the API for the given user input."""
        pass

class RAGAPIClient(APIClientInterface):
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize the API client.
        
        Args:
            base_url: Optional base URL for the API. If not provided, will use API_BASE_URL from .env
        """
        load_dotenv()
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        
    def get_response(self, user_input: str) -> str:
        """Get a response from the RAG API.
        
        Args:
            user_input: The user's input message
            
        Returns:
            str: The API response
            
        Note:
            This is a boilerplate implementation that returns a static response.
            In a real implementation, this would make an HTTP request to the API.
        """
        # TODO: Implement actual API call
        return "Test response"

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
