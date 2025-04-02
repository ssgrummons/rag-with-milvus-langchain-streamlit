from typing import List, Optional
from langchain_core.tools import tool
from langchain.schema import Document

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        The product of a and b
    """
    return a * b

@tool
def retrieve_context(query: str, top_k: int = 3) -> List[Document]:
    """Retrieve relevant context from Milvus vector store.
    
    Args:
        query: The search query.
        top_k: Number of documents to retrieve (default: 3).
        
    Returns:
        List of relevant documents with their content and metadata.
    """
    # TODO: Implement Milvus retrieval
    # This is a placeholder that will be implemented later
    return [
        Document(
            page_content="DataNinja supports the following authentication protocols: OAuth 2.0, API Key, and Basic Authentication.",
            metadata={"source": "placeholder"}
        )
    ] 