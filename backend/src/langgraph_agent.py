from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
import logging
from langchain_core.messages import AnyMessage
from models import DEFAULT_SYSTEM_PROMPT, OllamaModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[List[AnyMessage], add_messages]
    #tool_results: Annotated[Optional[Dict[str, Any]], "add_tool_results"]

def assistant(state: AgentState, tools: List[BaseTool], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dict[str, Any]:
    logger.info("Running assistant node")
    logger.info(f"Current messages: {[msg.type for msg in state['messages']]}")
    # Add system prompt if this is the first message
    messages = state["messages"]
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        messages.append(SystemMessage(content=system_prompt))

    # Append tool results as messages if present
    if state.get("tool_results"):
        for tool, result in state["tool_results"].items():
            messages.append(AIMessage(content=f"[Tool {tool} returned]: {result}"))
    
    # Create a chat model with tools
    chat_model = OllamaModelFactory().create_model()
    chat_with_tools = chat_model.bind_tools(tools)

    response = chat_with_tools.invoke(messages)
    logger.info(f"Assistant response: {getattr(response, 'content', '[no content]')}")

    return {
        "messages": [response],
    }

def create_agent_graph(tools: List[BaseTool], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("assistant", lambda state: assistant(state, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)

    return builder.compile()

def run_agent_graph(graph, user_message: str, system_prompt: Optional[str] = None) -> str:
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_results": None
    }

    logger.info(f"Starting graph with user message: {user_message}")
    final_state = graph.invoke(initial_state)

    final_messages = final_state["messages"]
    final_response = final_messages[-1]

    logger.info(f"Final response: {getattr(final_response, 'content', '[no content]')}")
    return final_response.content if hasattr(final_response, 'content') else str(final_response)

async def run_agent_graph_streaming(graph, user_message: str, system_prompt: Optional[str] = None):
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_results": None
    }

    logger.info(f"Starting streaming graph with user message: {user_message}")
    async for chunk in graph.astream(initial_state):
        if "messages" in chunk and chunk["messages"]:
            latest_message = chunk["messages"][-1]
            if hasattr(latest_message, 'content') and latest_message.content:
                logger.info(f"Streaming chunk: {latest_message.content}")
                yield latest_message.content
