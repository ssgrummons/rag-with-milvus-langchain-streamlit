from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import json
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

from models import get_model, bind_tools, handle_tool_call
from prompt_utils import build_messages
from tools import retrieve_context, multiply

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings."""
    # Milvus Configuration
    MILVUS_HOST: str = Field(..., description="Milvus service hostname")
    MILVUS_PORT: int = Field(..., description="Milvus service port")
    
    # Ollama Configuration
    OLLAMA_HOST: str = Field(..., description="Ollama service URL")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b", description="Default Ollama model to use")
    
    # RAG Configuration
    CHUNK_SIZE: int = Field(default=500, description="Size of text chunks for processing")
    CHUNK_OVERLAP: int = Field(default=100, description="Overlap between chunks")
    MAX_TOKENS: int = Field(default=2048, description="Maximum tokens for model output")
    TEMPERATURE: float = Field(default=0.7, description="Model temperature setting")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Security Configuration
    CORS_ORIGINS: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def allowed_origins(self) -> List[str]:
        """Get list of allowed CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    model: str
    system_prompt: Optional[str] = None
    user_prompt: str

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str

# Load settings
try:
    settings = Settings()
    # Configure logging based on settings
    logging.getLogger().setLevel(settings.LOG_LEVEL)
except Exception as e:
    logger.error(f"Failed to load settings: {str(e)}")
    raise

app = FastAPI(
    title="RAG Backend API",
    description="Backend API for RAG (Retrieval-Augmented Generation) application",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Hello World from RAG Backend!"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        # Add service health checks here
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration (excluding sensitive data)."""
    return {
        "milvus": {
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT
        },
        "ollama": {
            "host": settings.OLLAMA_HOST,
            "model": settings.OLLAMA_MODEL
        },
        "rag": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests with tool support.
    
    Args:
        request: The chat request containing model, system prompt, and user prompt.
        
    Returns:
        ChatResponse containing the model's response.
        
    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Get the model
        model = get_model(request.model)
        
        # Bind the tools
        model_with_tools = bind_tools(model, [multiply, retrieve_context])
        
        # Create messages list
        messages = []
        
        # Add system message if provided, otherwise use default
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        
        # Add user message
        messages.append(HumanMessage(content=request.user_prompt))
        
        # Handle the conversation with tool calls
        response = handle_tool_call(model_with_tools, messages, [multiply, retrieve_context])
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_tool_calls(
    model: Any,
    messages: List[BaseMessage],
    tools: List[Any]
) -> AsyncGenerator[str, None]:
    """Handle streaming responses with tool calls.
    
    Args:
        model: The chat model instance
        messages: List of messages in the conversation
        tools: List of available tools
        
    Yields:
        Response chunks as strings
    """
    try:
        # Get the initial response stream
        async for chunk in model.astream(messages):
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                # Handle tool calls
                for tool_call in chunk.tool_calls:
                    # Find the tool to execute
                    tool = next((t for t in tools if t.name == tool_call.name), None)
                    if tool:
                        # Execute the tool
                        tool_result = tool.invoke(tool_call.args)
                        
                        # Add the tool result to the conversation
                        messages.append(AIMessage(content="", tool_calls=[tool_call]))
                        messages.append(HumanMessage(content=f"Tool result: {tool_result}"))
                        
                        # Stream the final response
                        async for final_chunk in model.astream(messages):
                            if final_chunk.content:
                                yield final_chunk.content
            elif chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"Error in streaming tool calls: {str(e)}")
        yield f"Error: {str(e)}"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle streaming chat requests with tool support.
    
    Args:
        request: The chat request containing model, system prompt, and user prompt.
        
    Returns:
        StreamingResponse with the model's response chunks.
        
    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Get the model
        model = get_model(request.model)
        
        # Bind the tools
        model_with_tools = bind_tools(model, [multiply, retrieve_context])
        
        # Create messages list
        messages = []
        
        # Add system message if provided, otherwise use default
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        
        # Add user message
        messages.append(HumanMessage(content=request.user_prompt))
        
        async def stream_generator():
            async for chunk in handle_streaming_tool_calls(
                model_with_tools,
                messages,
                [multiply, retrieve_context]
            ):
                yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
