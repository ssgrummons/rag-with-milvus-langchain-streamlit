# RAG with Milvus, LangChain, and Streamlit

This repository demonstrates the construction of a basic Retrieval-Augmented Generation (RAG) system, integrating the following components:

- **Front-end Chat Interface**: Developed using Streamlit for user interactions.
- **Backend RAG Service**: Implemented with LangChain, connecting locally to Ollama and Milvus for processing and retrieval.
- **Milvus Vector Database**: Serves as the storage and retrieval system for vectorized document embeddings.
- **Attu Server**: Provides a graphical user interface for administering the Milvus database.
- **Document Processing Pipeline**: Handles parsing, chunking, vectorizing, and uploading markdown documents into Milvus.

All components are containerized and orchestrated using Docker Compose for streamlined deployment.

## Features

- **Interactive Chat Interface**: Engage with the system via a user-friendly Streamlit application.
- **Efficient Document Retrieval**: Leverage Milvus for high-performance vector similarity searches.
- **Seamless Integration**: Utilize LangChain to connect the front-end with backend services, ensuring smooth data flow.
- **Administrative Control**: Manage and monitor the Milvus database using the Attu GUI.
- **Automated Document Processing**: Automatically parse, chunk, vectorize, and store markdown documents into Milvus.

## Deployment

### Running Ollama

First, install [Ollama](https://ollama.com) on Mac, Linux, or Windows. If you have a local GPU, this will allow you to leverage it for improved performance. Once installed, download and run the project's default [qwen2:7b](https://ollama.com/library/qwen2:7b) model:

```bash
ollama pull qwen2:7b
```

You can validate that Ollama is running by navigating to [http://localhost:11434](http://localhost:11434). Your Docker containers will access Ollama on the host using `http://host.docker.internal:11434`.

**For the system to work, you need a model that supports [tool use](https://ollama.com/search?c=tools).** You can configure different models in the backend environment. During development, some models handled different tools better than others. Experiment with different models and update the [backend `.env` file`](backend/src/.env-template) accordingly.

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

   ```bash
   docker-compose -f ./deployment/docker-compose.yml up -d --build
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

For knowledge base ingestion, refer to the [Pipeline Documentation](pipelines/README.md).

## Components Overview

- **Streamlit Application**: Located in the `frontend` directory, this component provides the chat interface.
- **LangChain Backend**: Found in the `backend` directory, it manages the RAG process and interacts with both the Streamlit front-end and the Milvus database.
- **Milvus Vector Database**: Configured in the `deployment` directory, it stores vectorized document representations. Storage for Milvus resides in `deployment/vols` with separate volumes for `etcd`, `minio`, and `milvus`.
- **Attu Server**: Also configured in `deployment`, offering a GUI for Milvus management.
- **Document Processing Pipeline**: Contained in the `pipelines` directory, this component automates markdown document ingestion.
- **Fake Documentation**: The `docs` folder contains fictional technical documentation for a made-up SaaS solution called *DataNinja*. After processing this documentation into Milvus, you can query it for *DataNinja* setup, administration, and usage guidance.

## Prerequisites

Ensure you have the following installed:

- **Ollama**: For serving local models.
- **Docker**: For containerization and deployment.
- **Docker Compose**: To orchestrate the multi-container deployment.

## Notes

This project is intended as a proof-of-concept, demonstrating the foundational aspects of a RAG system and the integration of its various components.

For further technical reference:

- **Milvus Installation**: [Milvus documentation](https://milvus.io/docs/install_standalone-docker-compose.md)
- **Attu Installation**: [Attu Docker Installation Guide](https://milvus.io/docs/v2.2.x/attu_install-docker.md)
- **LangChain Integration**: [LangChain-Milvus](https://github.com/langchain-ai/langchain-milvus)

By following this guide, you can set up a basic RAG system that integrates Ollama, Streamlit, LangChain, Milvus, and Attu—providing a foundation for further development and experimentation.

