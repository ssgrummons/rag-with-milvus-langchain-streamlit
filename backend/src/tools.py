from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.schema import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
TOP_K = int(os.getenv("TOP_K", "3"))

class EmbeddingService:
    """Service for handling text embeddings."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not self.model:
            self._initialize_embeddings()
        
        try:
            # Use sentence_transformers to generate embeddings
            embeddings = self.model.encode(texts)
            # Convert numpy arrays to lists for compatibility
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to get embeddings: {str(e)}")
            raise

class MilvusService:
    """Service for interacting with Milvus vector database."""
    
    def __init__(self, host: str = MILVUS_HOST, port: str = MILVUS_PORT):
        """Initialize the Milvus service.
        
        Args:
            host: Milvus host
            port: Milvus port
        """
        self.host = host
        self.port = port
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Milvus."""
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            connections.disconnect()
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {str(e)}")
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a Milvus collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Milvus collection
            
        Raises:
            ValueError: If the collection does not exist
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        return Collection(collection_name)
    
    def search(self, collection_name: str, query_embedding: List[float], 
           top_k: int = TOP_K, search_field: str = "embedding", 
           output_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Query embedding vector
            top_k: Number of results to return
            search_field: Field to search on
            output_fields: Fields to return in results
            
        Returns:
            List of search results
        """
        try:
            collection = self.get_collection(collection_name)
            collection.load()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[query_embedding],
                anns_field=search_field,
                param=search_params,
                limit=top_k,
                output_fields=output_fields or ["content", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "content": getattr(hit, "content", ""),  # Fix: use getattr instead of .get()
                        "metadata": getattr(hit, "metadata", {})  # Fix: use getattr instead of .get()
                    }
                    formatted_results.append(result)
            
            collection.release()
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search Milvus: {str(e)}")
            raise


class ContextRetriever:
    """Service for retrieving context from Milvus."""
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        """Initialize the context retriever.
        
        Args:
            collection_name: Name of the Milvus collection
        """
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()
        self.milvus_service = MilvusService()
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve context for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of context items
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_service.get_embeddings([query])[0]
            
            # Search Milvus
            results = self.milvus_service.search(
                self.collection_name, 
                query_embedding, 
                top_k=top_k
            )
            
            # Format results
            context_items = []
            for result in results:
                context_items.append({
                    "text": result["content"],  # Map content to text for compatibility
                    "metadata": result["metadata"]
                })
            
            return context_items
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}")
            raise

# Initialize the context retriever
context_retriever = ContextRetriever()

@tool
def retrieve_context(query: str) -> List[Document]:
    """Retrieve relevant context about DataNinja from the knowledge base. The knowledge base is a Milvus vector store.  Any queries about DataNinja should be answered using this tool.
    
    Args:
        query: The search query.
        
    Returns:
        List of the content of the relevant documents with their content and metadata.
    """
    top_k = TOP_K
    try:
        # Get context from the retriever
        context_items = context_retriever.retrieve(query, top_k)
        
        # Convert to Document objects
        documents = []
        for item in context_items:
            documents.append(
                Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
            )
        
        return documents
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of the two numbers
    """
    return a * b 