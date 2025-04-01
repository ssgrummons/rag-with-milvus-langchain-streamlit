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

For detailed instructions navigate to the [Deployment Documentation](deployment/README.md).

## Components Overview

- **Streamlit Application**: Located in the `frontend` directory, this component provides the chat interface for user interactions.
- **LangChain Backend**: Found in the `backend` directory, it manages the RAG process, interfacing with both the Streamlit front-end and the Milvus database.
- **Milvus Vector Database**: Configured within the `deployment` directory, it stores vector representations of the processed documents.
- **Attu Server**: Also configured in the `deployment` directory, offering a GUI for Milvus database management.
- **Document Processing Pipeline**: Contained in the `pipelines` directory, this component automates the ingestion of markdown documents into the system.

## Prerequisites

Ensure the following are installed on your system:

- **Docker**: For containerization and deployment.
- **Docker Compose**: To orchestrate the multi-container deployment.

## Notes

This project is intended for proof-of-concept purposes, illustrating the foundational aspects of a RAG system and the integration of its various components.

For more detailed information on the technologies used:

- **Milvus Installation**: Refer to the [Milvus documentation](https://milvus.io/docs/install_standalone-docker-compose.md) for guidance on setting up Milvus with Docker Compose.
- **Attu Installation**: Instructions for installing Attu with Docker can be found [here](https://milvus.io/docs/v2.2.x/attu_install-docker.md).
- **LangChain Integration**: Explore the [LangChain-Milvus integration](https://github.com/langchain-ai/langchain-milvus) for more details on connecting LangChain with Milvus.

By following this guide, you can set up a basic RAG system that showcases the integration of Streamlit, LangChain, Milvus, and Attu, providing a foundation for further development and experimentation. 