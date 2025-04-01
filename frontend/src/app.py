import streamlit as st
import os
from api_client import get_response
from dotenv import load_dotenv
from typing import List, Dict

class ChatApp:
    def __init__(self):
        load_dotenv()
        self.port = int(os.getenv("STREAMLIT_PORT", 8501))
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
        
        # Get response from API client
        response = get_response(user_input)
        
        # Add response to session state
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    def run(self) -> None:
        """Run the chat application."""
        st.title("Chat with the RAG Model")
        
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
