from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
import logging
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
        logger.info(f"Adding System Prompt")
        messages.append(SystemMessage(content=system_prompt))
    
    # Create a chat model with tools
    chat_model = OllamaModelFactory().create_model()
    chat_with_tools = chat_model.bind_tools(tools)

    response = chat_with_tools.invoke(messages)
    logger.info(f"Assistant response: {getattr(response, 'content', '[no content]')}")

    return {
        "messages": [response],
    }

def final_answer(state: AgentState) -> AgentState:
    logger.info("Processing Final Answer")
    messages = state["messages"]
    follow_up = SystemMessage(
            content="""
            You are a helpful AI assistant who is responsible for reviewing all the content in the thread and providing a final answer to the initial human question.
            You are not allowed to call any tools any more you already have the proper context provided by those tools.
            Review the human question, the system prompt, and the information provided by the tools.
            Provide a final response in natural language.  This is important: Even if your source data or tool responses are in structured format, your job is to translate that into **clear, complete, natural sentences** in the final step.
            """
        )
    messages = messages + [follow_up]
    model_name = "granite3.2"
    chat_model = OllamaModelFactory().create_model(model_name, format=None)
    response = chat_model.invoke(messages)
    logger.info(f"Assistant response: {getattr(response, 'content', '[no content]')}")
    return {
        "messages": [response],
    }

def create_agent_graph(tools: List[BaseTool], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("assistant", lambda state: assistant(state, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("final_answer", final_answer)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", "final_answer")
    builder.add_edge("final_answer", END)

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
