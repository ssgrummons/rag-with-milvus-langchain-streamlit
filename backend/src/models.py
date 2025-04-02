from langchain_community.chat_models import ChatOllama

def get_model(model_name: str = "llama3"):
    return ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",  # Ollama default endpoint
        temperature=0.7,
    )
