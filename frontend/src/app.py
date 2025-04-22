import streamlit as st
import os
import asyncio
from api_client import get_response, get_streaming_response
from dotenv import load_dotenv
from typing import List, Dict, Optional

class ChatApp:
    def __init__(self, use_streaming: bool = True):
        load_dotenv()
        self.port = int(os.getenv("STREAMLIT_PORT", 8501))
        self.use_streaming = use_streaming
        self._initialize_session_state()
        self._setup_page_config()

    def _initialize_session_state(self) -> None:
        """Initialize the chat history in session state."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def _setup_page_config(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(page_title="Chat with RAG", layout="wide")

    def _display_chat_history(self) -> None:
        """Display all messages from the chat history."""
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_user_input(self, user_input: str) -> None:
        """Handle user input and generate response."""
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            if self.use_streaming:
                # Use streaming response
                self._handle_streaming_response(user_input, response_placeholder)
            else:
                # Use non-streaming response (backward compatibility)
                response = get_response(user_input)
                response_placeholder.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
    
    def _handle_streaming_response(self, user_input: str, placeholder) -> None:
        """Handle streaming response from the API."""
        # Create a container for the streaming response
        full_response = ""
        
        # Define a callback to update the placeholder
        def update_placeholder(chunk: str) -> None:
            nonlocal full_response
            full_response += chunk
            placeholder.markdown(full_response)
        
        # Run the async streaming function
        async def run_streaming():
            async for chunk in get_streaming_response(user_input, update_placeholder):
                pass  # The callback handles the updates
        
        # Execute the async function
        asyncio.run(run_streaming())
        
        # Add the complete response to session state
        st.session_state["messages"].append({"role": "assistant", "content": full_response})

    def run(self) -> None:
        """Run the chat application."""
        st.title("Chat with the RAG Model")
        
        # Add a toggle for streaming mode
        self.use_streaming = st.sidebar.checkbox("Use streaming responses", value=True)
        
        # Display chat history
        self._display_chat_history()
        
        # Handle user input
        if user_input := st.chat_input("Type your message..."):
            self._handle_user_input(user_input)

def main() -> None:
    """Main function to run the chat application."""
    chat_app = ChatApp()
    chat_app.run()

if __name__ == "__main__":
    main()
