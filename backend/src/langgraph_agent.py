from typing import Annotated, TypedDict, List, Dict, Any, AsyncGenerator
from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
import logging
import json
from models import OllamaModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict, total=False):
    """State for the agent graph."""
    messages: Annotated[List[AnyMessage], add_messages]
    tool_results: Any
    streaming: bool
    streamed_output: AsyncGenerator[str, None]

def assistant(state: AgentState, tools: List[BaseTool], system_prompt: str) -> Dict[str, Any]:
    logger.info("Running assistant node")
    logger.info(f"Current messages: {[msg.type for msg in state['messages']]}")
    messages = state["messages"]
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        logger.info(f"Adding System Prompt")
        messages.append(SystemMessage(content=system_prompt))
    
    chat_model = OllamaModelFactory().create_model(format=None)
    chat_with_tools = chat_model.bind_tools(tools)
    response = chat_with_tools.invoke(messages)
    logger.info(f"Assistant response: {getattr(response, 'content', '[no content]')}")
    return {"messages": [response]}

async def final_answer(state: AgentState) -> Dict[str, Any]:
    logger.debug("Running final answer node")
    messages = state["messages"]
    messages.append(AIMessage(content="Finalizing answer..."))
    streaming = state.get("streaming", False)

    chat_model = OllamaModelFactory().create_model(format=None)

    if streaming:
        logger.debug("Running agent graph with streaming...")
        stream = chat_model.astream(messages)

        async def wrapped_stream():
            logger.debug("Running streaming wrapper...")
            async for chunk in stream:
                logger.debug(f"Streamed chunk: {chunk}")
                if hasattr(chunk, "content"):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk

        return {"streamed_output": wrapped_stream()}
    else:
        logger.debug("Running agent graph without streaming...")
        response = chat_model.invoke(messages)
        return {"messages": [response]}


def create_agent_graph(tools: List[BaseTool], system_prompt: str) -> StateGraph:
    builder = StateGraph(AgentState)
    
    # Define nodes
    builder.add_node("assistant", lambda s: assistant(s, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("final_answer", final_answer)  # This node handles streaming
    
    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "final_answer")
    builder.add_edge("final_answer", END)
    
    return builder.compile() 

def run_agent_graph(graph, initial_state: dict) -> AnyMessage:
    initial_state["streaming"] = False  # Ensure streaming is off
    final_state = graph.invoke(initial_state)
    return final_state["messages"][-1]

async def run_agent_graph_streaming(
    graph,
    initial_state: dict
) -> AsyncGenerator[str, None]:
    initial_state["streaming"] = True
    try:
        logger.debug("Running the Agent Graph Streaming Function...")
        final_state = await graph.ainvoke(initial_state)  # Run full graph
        stream = final_state.get("streamed_output", None)
        logger.debug(f'Debug Stream: {stream}')

        if hasattr(stream, "__aiter__"): 
            async for chunk in stream:
                logger.debug(f"Debug Chunk: {chunk}")
                yield chunk
        else:
            yield "No stream available."
    except Exception as e:
        logger.exception("Error during agent graph streaming")
        yield f"Error: {str(e)}"
