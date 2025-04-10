from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict, AsyncGenerator, Union
from pydantic_settings import BaseSettings
from langchain_ollama import ChatOllama
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import BaseTool

class OllamaSettings(BaseSettings):
    """Settings for Ollama model configuration."""
    OLLAMA_HOST: str
    OLLAMA_MODEL: str 
    MAX_TOKENS: int 
    TEMPERATURE: float 

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Default system prompt for the ReAct framework
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to help answer questions.
When you need to perform calculations or retrieve information, use the available tools.
For mathematical questions, use the multiply tool to get accurate results.
After using a tool, always provide a final response in natural language. Explain how you arrived at the result clearly, step by step.

YOU MUST ALWAYS produce final answers as **natural, narrative language**. 
NEVER use JSON, XML, YAML, or any structured format in the final response.

Do NOT return answers in list, dictionary, or code block formats.
You should speak as if you are explaining something to a human in plain English.

❌ BAD Final Answer Example:
{
  "email": "support@dataninja.com",
  "phone": "1-800-DATA-NINJA"
}

✅ GOOD Final Answer Example:
You can contact DataNinja support by email at support@dataninja.com, or call 1-800-DATA-NINJA anytime.

This is important: Even if your source data or tool responses are in structured format, your job is to translate that into **clear, complete, natural sentences** in the final step.

Available tools:
- multiply: Multiply two numbers together
- retrieve_context: Get any information about DataNinja from the knowledge base

When using tools:
1. Think about which tool would be most appropriate
2. Use the tool with the correct parameters
3. Explain the result in a clear way

Remember to use tools when they provide more accurate or helpful results.

Follow the ReAct framework:
1. Thought: Think about what you need to do
2. Action: Use a tool if needed
3. Observation: Observe the result
4. Response: Provide a final answer

Here are the rules you should always follow:
1. Always explain your reasoning before using a tool
2. Use only the tools that are available to you
3. Always use the right arguments for the tools
4. Take care to not chain too many sequential tool calls in the same response
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters
6. Don't give up! You're in charge of solving the task
"""

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
        # Custom system prompt for tool usage
        system_prompt = """You are a helpful AI assistant that can use tools to help answer questions.
When you need to perform calculations or retrieve information, use the available tools.
For mathematical questions, use the multiply tool to get accurate results.
Always explain your reasoning and show your work when using tools.

Available tools:
- multiply: Multiply two numbers together
- retrieve_context: Get relevant information from the knowledge baseRetrieve relevant context about DataNinja from the knowledge base. The knowledge base is a Milvus vector store.  Any queries about DataNinja should be answered using this tool.

When using tools:
1. Think about which tool would be most appropriate
2. Use the tool with the correct parameters
3. Explain the result in a clear way

Remember to use tools when they would provide more accurate or helpful results than trying to calculate or recall information yourself."""

        # Ensure model name doesn't have trailing spaces
        model = (model_name or self.settings.OLLAMA_MODEL).strip()

        return ChatOllama(
            model=model,
            base_url=self.settings.OLLAMA_HOST,
            temperature=self.settings.TEMPERATURE,
            num_predict=self.settings.MAX_TOKENS,
            system=system_prompt,
            format="json",  # Ensure JSON mode for tool calling
            callbacks=None,  # Disable callbacks to handle tool calls manually
            verbose=True
        )

class ToolHandler(ABC):
    """Abstract base class for tool handling."""
    
    @abstractmethod
    def handle_tool_call(self, model: BaseChatModel, messages: List[BaseMessage], tools: List[BaseTool]) -> Any:
        """Handle a tool call and return the response."""
        pass

class StreamingToolHandler(ToolHandler):
    """Handler for streaming tool calls."""
    
    async def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming tool calls and yield responses."""
        try:
            # Get the initial response stream
            async for chunk in model.astream(messages):
                # Handle content chunks
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    continue
                
                # Handle tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        # Find the tool to execute
                        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                        tool = next((t for t in tools if t.name == tool_name), None)
                        
                        if tool:
                            # Get the arguments
                            args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                            
                            # Execute the tool
                            tool_result = tool.invoke(args)
                            
                            # Add the tool result to the conversation
                            messages.append(SystemMessage(content=f"""You have just ran the tool. The tools provided the context you need to answer the user's question.
                                                          Now, respond with a detailed answer to the user's question based on this information.  
                                                          Your answer should be in natural language.  Do not use any more tools.
                                                          Do not make up any information.  
                                                          Only use the context provided by the tool to answer the question accurately and truthfully.
                                                          Context: {tool_result}"""))
                            
                            # Stream the final response
                            async for final_chunk in model.astream(messages):
                                if hasattr(final_chunk, 'content') and final_chunk.content:
                                    yield final_chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

class NonStreamingToolHandler(ToolHandler):
    """Handler for non-streaming tool calls."""
    
    def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> str:
        """Handle non-streaming tool calls and return the final response."""
        # Get the initial response from the model
        response = model.invoke(messages)

        # If there are tool calls, execute them and continue the conversation
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                # Find the tool to execute
                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                tool = next((t for t in tools if t.name == tool_name), None)
                
                if tool:
                    # Get the arguments
                    args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                    
                    # Execute the tool
                    tool_result = tool.invoke(args)
                    
                    # Add the tool result to the conversation
                    messages.append(SystemMessage(content=f"The result of the {tool_name} operation is: {tool_result}. Now, please respond with a detailed answer and explanation in natural language without calling any tools."))
            
            # Get the final response
            final_response = model.invoke(messages)
            return final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        # If no tool calls, return the content from the initial response
        return response.content if hasattr(response, 'content') else str(response)

class ModelManager:
    """Manages chat model instances and their lifecycle."""
    
    def __init__(self, factory: Optional[ModelFactory] = None):
        """Initialize the model manager with a factory."""
        self.factory = factory or OllamaModelFactory()
        self.streaming_handler = StreamingToolHandler()
        self.non_streaming_handler = NonStreamingToolHandler()
    
    def get_model(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Get a chat model instance.
        
        Args:
            model_name: Optional model name to override the default.
            
        Returns:
            A configured chat model instance.
        """
        return self.factory.create_model(model_name)

    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """Bind tools to a chat model.
        
        Args:
            model: The chat model to bind tools to.
            tools: List of tools to bind.
            
        Returns:
            The chat model with tools bound.
        """
        return model.bind_tools(tools)

    async def handle_streaming_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming tool calls."""
        return self.streaming_handler.handle_tool_call(model, messages, tools)

    def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> str:
        """Handle non-streaming tool calls."""
        return self.non_streaming_handler.handle_tool_call(model, messages, tools)

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

def bind_tools(model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
    """Bind tools to a chat model (backward compatibility function).
    
    Args:
        model: The chat model to bind tools to.
        tools: List of tools to bind.
        
    Returns:
        The chat model with tools bound.
    """
    return _model_manager.bind_tools(model, tools)

def handle_tool_call(model: BaseChatModel, messages: List[BaseMessage], tools: List[BaseTool]) -> str:
    """Handle non-streaming tool calls (backward compatibility function).
    
    Args:
        model: The chat model instance.
        messages: List of messages in the conversation.
        tools: List of available tools.
        
    Returns:
        The final response from the model.
    """
    return _model_manager.handle_tool_call(model, messages, tools)

async def handle_streaming_tool_call(
    model: BaseChatModel,
    messages: List[BaseMessage],
    tools: List[BaseTool]
) -> AsyncGenerator[str, None]:
    """Handle streaming tool calls (backward compatibility function).
    
    Args:
        model: The chat model instance.
        messages: List of messages in the conversation.
        tools: List of available tools.
        
    Returns:
        An asynchronous generator yielding responses.
    """
    async for chunk in await _model_manager.handle_streaming_tool_call(model, messages, tools):
        yield chunk
