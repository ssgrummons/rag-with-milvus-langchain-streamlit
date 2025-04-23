# Local Agentic RAG with Milvus, LangGraph, and Streamlit

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
4. Assistant → Final Answer (to summarize the results in natural language)

## Prerequisites

Ensure you have the following installed:

- **Ollama**: For serving local models.
- **Docker**: For containerization and deployment.
- **Docker Compose**: To orchestrate the multi-container deployment.

## Setup and Installation

### Running Ollama

First, install [Ollama](https://ollama.com) on Mac, Linux, or Windows. If you have a local GPU, this will allow you to leverage it for improved performance. Once installed, download and run the project's default [qwen2:7b](https://ollama.com/library/qwen2:7b) model:

```bash
ollama pull qwen2:7b
```

You can validate that Ollama is running by navigating to [http://localhost:11434](http://localhost:11434). Your Docker containers will access Ollama on the host using `http://host.docker.internal:11434`.

**For the system to work, you need a model that supports [tool use](https://ollama.com/search?c=tools).** You can configure different models in the backend environment. During development, some models handled different tools better than others. Experiment with different models and update the [backend .env file`](backend/src/.env-template) accordingly.  I've found similar performance with `granite3.2`.

### Preparing your Knowledge Base

For knowledge base ingestion, refer to the [Pipeline Documentation](pipelines/README.md).

### Deploying Services

All services except for the document processing pipeline can be started using [Docker Compose](https://docs.docker.com/compose/). The [docker-compose.yml](deployment/docker-compose.yml) file is configured to use `.env` files for environment variables.

#### Prepare the Environment Files

Before starting the containers, create `.env` files based on the provided templates:

1. **Backend Environment Variables**:
   - Copy [backend/src/.env-template](backend/src/.env-template) and save it as `.env` in the same directory.
   - Add your [LangSmith](https://www.langchain.com/langsmith) settings if applicable. Otherwise, leave the defaults.

2. **Frontend Environment Variables**:
   - Copy [frontend/src/.env-template](frontend/src/.env-template) and save it as `.env` in the same directory.
   - No changes are necessary.

#### Start the System

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ssgrummons/rag-with-milvus-langchain-streamlit.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd rag-with-milvus-langchain-streamlit
   ```

3. **Start the Services**:

   The `EMBEDDING_MODEL` environment variable is used in both the runtime and the build environment.  In order to define it once, we will pull it from the `.env` file before we run the Docker Compose command.  This enables us define the environment varialbe once and use it throughout the system.  You can do this by exporting the environment variable:

   ```bash
   export EMBEDDING_MODEL=$(grep EMBEDDING_MODEL ./backend/src/.env | cut -d '=' -f2-) | docker-compose -f ./deployment/docker-compose.yml up -d --build
   ```

   This command builds and starts all the services in detached mode.

4. **Access the Applications**:

   - **Streamlit Interface**: Navigate to [http://localhost:8501](http://localhost:8501).
   - **Attu GUI**: Open [http://localhost:8088](http://localhost:8088) and connect to Milvus with:
     - **Milvus Address**: `milvus-standalone:19530`
     - **Milvus Database**: `default`
     - **Authentication**: None
     - **Enable SSL**: Unchecked
   - **Backend API**: View the OpenAPI documentation at [http://localhost:8000/docs#/](http://localhost:8000/docs#/).

For more details, check the [Deployment Documentation](deployment/README.md).

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

