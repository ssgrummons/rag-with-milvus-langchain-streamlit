from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional, List
import json

from models import get_model
from prompt_utils import build_messages


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
    OLLAMA_MODEL: str = Field(default="deepseek-r1:7b", description="Default Ollama model to use")
    
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

    @property
    def allowed_origins(self) -> List[str]:
        """Get list of allowed CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

class ChatRequest(BaseModel):
    """Chat request model."""
    model: Optional[str] = Field(default="deepseek-r1:7b", description="Model to use for chat")
    system_prompt: Optional[str] = Field(
        default="You are a helpful AI assistant.",
        description="System prompt for the chat"
    )
    user_prompt: str = Field(..., description="User's input prompt")

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

@app.post("/chat")
async def chat(req: ChatRequest):
    llm = get_model(req.model)
    messages = build_messages(req.system_prompt, req.user_prompt)
    response = llm.invoke(messages)
    return JSONResponse({"response": response.content})

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    llm = get_model(req.model)
    messages = build_messages(req.system_prompt, req.user_prompt)
    def stream_generator():
        for chunk in llm.stream(messages):
            yield chunk.content or ""
    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
