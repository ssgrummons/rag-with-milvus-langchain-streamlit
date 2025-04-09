from typing import Annotated, TypedDict, List, Dict, Any, Optional, Union, cast
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import logging
from langchain_core.messages import AnyMessage
from models import DEFAULT_SYSTEM_PROMPT, OllamaModelFactory
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[List[AnyMessage], "add_messages"]
    tool_results: Annotated[Optional[Dict[str, Any]], "add_tool_results"]



def assistant(state: AgentState, tools: List[BaseTool], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dict[str, Any]:
    """Assistant node that processes messages and decides whether to use tools.
    
    Args:
        state: The current state of the agent.
        tools: List of available tools.
        system_prompt: System prompt to use.
        
    Returns:
        Updated state.
    """
    # If this is the first message, add the system prompt
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        # Add system message with the ReAct framework prompt
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
    else:
        messages = state["messages"]

    if state.get("tool_results"):
        for tool, result in state["tool_results"].items():
            messages.append(AIMessage(content=f"[Tool {tool} returned]: {result}"))
    
    # Create a chat model with tools
    chat_model = OllamaModelFactory().create_model()
    chat_with_tools = chat_model.bind_tools(tools)
    
    return {
        "messages": [chat_with_tools.invoke(messages)],
    }

def create_agent_graph(tools: List[BaseTool], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> StateGraph:
    """Create the agent graph.
    
    Args:
        tools: List of available tools.
        system_prompt: System prompt to use.
        
    Returns:
        The compiled agent graph.
    """
    # Create the state graph
    builder = StateGraph(AgentState)
    
    # Define nodes
    builder.add_node("assistant", lambda state: assistant(state, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    
    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)
    # Compile the graph
    return builder.compile()

def run_agent_graph(graph, user_message: str, system_prompt: Optional[str] = None) -> str:
    """Run the agent graph with a user message.
    
    Args:
        graph: The compiled agent graph.
        user_message: The user message.
        system_prompt: Optional system prompt to use.
        
    Returns:
        The final response.
    """
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_results": None
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Extract the final response
    final_messages = final_state["messages"]
    final_response = final_messages[-1]
    
    # Return the content of the final response
    return final_response.content if hasattr(final_response, 'content') else str(final_response)

async def run_agent_graph_streaming(graph, user_message: str, system_prompt: Optional[str] = None):
    """Run the agent graph with a user message in streaming mode.
    
    Args:
        graph: The compiled agent graph.
        user_message: The user message.
        system_prompt: Optional system prompt to use.
        
    Yields:
        Chunks of the response.
    """
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_results": None
    }
    
    # Run the graph in streaming mode
    async for chunk in graph.astream(initial_state):
        if "messages" in chunk and chunk["messages"]:
            latest_message = chunk["messages"][-1]
            if hasattr(latest_message, 'content') and latest_message.content:
                yield latest_message.content 