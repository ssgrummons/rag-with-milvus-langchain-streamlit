# RAG Application with Milvus, LangChain, and Streamlit

This project implements a Retrieval-Augmented Generation (RAG) application using Milvus for vector storage, LangChain for LLM interactions, and Streamlit for the frontend. The application now uses LangGraph to implement a ReAct framework for tool usage.

## Architecture

The application is structured as follows:

### Backend

- **FastAPI**: Provides REST API endpoints for chat and streaming
- **LangGraph**: Implements a ReAct framework with a graph-based architecture
- **Milvus**: Vector database for storing and retrieving embeddings
- **Ollama**: Local LLM service for generating responses
- **Tools**: Custom tools for retrieving context and performing calculations

### Frontend

- **Streamlit**: Web interface for interacting with the backend
- **API Client**: Handles communication with the backend API

## LangGraph Implementation

The application now uses LangGraph to implement a ReAct framework with the following components:

1. **AgentState**: Defines the state of the agent, including messages and tool results
2. **Assistant Node**: Processes messages and decides whether to use tools
3. **Tool Node**: Executes tools when needed
4. **Graph**: Connects the nodes with conditional edges

The graph flow is as follows:
1. Start → Assistant
2. Assistant → Tools (if tools are needed) or End (if no tools are needed)
3. Tools → Assistant (to continue the conversation)
4. Assistant → End (when the conversation is complete)

## API Endpoints

- `/chat`: Non-streaming chat endpoint using LangGraph
- `/chat/stream`: Streaming chat endpoint using LangGraph
- `/chat/legacy`: Legacy non-streaming chat endpoint (for backward compatibility)
- `/chat/stream/legacy`: Legacy streaming chat endpoint (for backward compatibility)

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Set up environment variables in `.env` file
4. Start the backend:
   ```bash
   poetry run python -m backend.src.app
   ```
5. Start the frontend:
   ```bash
   poetry run streamlit run frontend/src/app.py
   ```

## Environment Variables

- `MILVUS_HOST`: Milvus service hostname
- `MILVUS_PORT`: Milvus service port
- `OLLAMA_HOST`: Ollama service URL
- `OLLAMA_MODEL`: Default Ollama model to use
- `EMBEDDING_MODEL`: Embedding model to use
- `COLLECTION_NAME`: Milvus collection name
- `TOP_K`: Number of results to retrieve from Milvus

## Tools

The application includes the following tools:

1. **retrieve_context**: Retrieves relevant context from the Milvus vector store
2. **multiply**: Multiplies two numbers together

## ReAct Framework

The application uses the ReAct framework for tool usage:

1. **Thought**: The assistant thinks about what it needs to do
2. **Action**: The assistant uses a tool if needed
3. **Observation**: The assistant observes the result
4. **Response**: The assistant provides a final answer

## License

MIT

