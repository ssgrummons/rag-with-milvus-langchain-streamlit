from abc import ABC, abstractmethod
from typing import Optional
from pydantic_settings import BaseSettings
from langchain_ollama import ChatOllama
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage

class OllamaSettings(BaseSettings):
    """Settings for Ollama model configuration."""
    OLLAMA_HOST: str
    OLLAMA_MODEL: str 
    MAX_TOKENS: int 
    TEMPERATURE: float 

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # This tells Pydantic to ignore extra fields

class ModelFactory(ABC):
    """Abstract factory for creating chat models."""
    
    @abstractmethod
    def create_model(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Create a chat model instance."""
        pass

class OllamaModelFactory(ModelFactory):
    """Factory for creating Ollama chat models."""
    
    def __init__(self, settings: Optional[OllamaSettings] = None):
        """Initialize the factory with settings."""
        self.settings = settings or OllamaSettings()
    
    def create_model(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Create an Ollama chat model instance.
        
        Args:
            model_name: Optional model name to override the default from settings.
            
        Returns:
            A configured ChatOllama instance.
        """
        return ChatOllama(
            model=model_name or self.settings.OLLAMA_MODEL,
            base_url=self.settings.OLLAMA_HOST,
            temperature=self.settings.TEMPERATURE,
            num_predict=self.settings.MAX_TOKENS,
        )

class ModelManager:
    """Manages chat model instances and their lifecycle."""
    
    def __init__(self, factory: Optional[ModelFactory] = None):
        """Initialize the model manager with a factory."""
        self.factory = factory or OllamaModelFactory()
    
    def get_model(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Get a chat model instance.
        
        Args:
            model_name: Optional model name to override the default.
            
        Returns:
            A configured chat model instance.
        """
        return self.factory.create_model(model_name)

# Create a singleton instance for backward compatibility
_model_manager = ModelManager()

def get_model(model_name: Optional[str] = None) -> BaseChatModel:
    """Get a chat model instance (backward compatibility function).
    
    Args:
        model_name: Optional model name to override the default.
        
    Returns:
        A configured chat model instance.
    """
    return _model_manager.get_model(model_name)
