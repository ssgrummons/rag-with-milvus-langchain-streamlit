from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import json
import yaml
import os

from tools import retrieve_context, multiply
from langgraph_agent import create_agent_graph, run_agent_graph, run_agent_graph_streaming

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompts from YAML file
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_path = os.path.join(current_dir, "prompts.yaml")
with open(prompts_path, 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Get the system prompt from the YAML file
assistant_system_prompt = prompt_templates["assistant_system_prompt"]

class Settings(BaseSettings):
    """Application settings."""
    # Milvus Configuration
    MILVUS_HOST: str = Field(..., description="Milvus service hostname")
    MILVUS_PORT: int = Field(..., description="Milvus service port")
    
    # Ollama Configuration
    OLLAMA_HOST: str = Field(..., description="Ollama service URL")
    OLLAMA_MODEL: str = Field(default="qwen2:7b", description="Default Ollama model to use")
    
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
    model: Optional[str] = "qwen2:7b"  # Default model
    system_prompt: Optional[str] = None  # Use default from langgraph_agent
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

# Initialize the agent graph
tools = [multiply, retrieve_context]

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
    agent_graph = create_agent_graph(tools, assistant_system_prompt, False)
    try:
        # Use the LangGraph agent
        response = run_agent_graph(agent_graph, request.user_prompt)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle streaming chat requests with tool support."""
    agent_graph = create_agent_graph(tools, assistant_system_prompt, True)
    try:
        # Use the LangGraph agent in streaming mode
        
        async def stream_generator():
            try:
                async for chunk in run_agent_graph_streaming(agent_graph, request.user_prompt):
                    if chunk:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # Send an end-of-stream marker
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
