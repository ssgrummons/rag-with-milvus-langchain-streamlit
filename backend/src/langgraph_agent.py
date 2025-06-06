from typing import Annotated, TypedDict, List, Dict, Any, AsyncGenerator
from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
import logging
from models import OllamaModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict, total=False):
    """State for the agent graph."""
    messages: Annotated[List[AnyMessage], add_messages]
    streamed_output: AsyncGenerator[str, None]

def assistant(state: AgentState, tools: List[BaseTool], system_prompt: str) -> Dict[str, Any]:
    logger.info("Running assistant node")
    logger.info(f"Current messages: {[msg.type for msg in state['messages']]}")
    messages = state["messages"]
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        logger.info("Adding system prompt to message history")
        messages.append(SystemMessage(content=system_prompt))
    
    chat_model = OllamaModelFactory().create_model(format=None)
    chat_with_tools = chat_model.bind_tools(tools)
    response = chat_with_tools.invoke(messages)

    logger.info(f"Assistant response: {getattr(response, 'content', '[no content]')}")
    return {"messages": [response]}

async def final_answer_async(state: AgentState) -> Dict[str, Any]:
    logger.debug("Running final answer async (streaming) node")
    messages = state["messages"]
    messages.append(AIMessage(content="Finalizing answer..."))

    chat_model = OllamaModelFactory().create_model(format=None)
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

def final_answer_sync(state: AgentState) -> Dict[str, Any]:
    logger.debug("Running final answer sync node")
    messages = state["messages"]
    messages.append(AIMessage(content="Finalizing answer..."))

    chat_model = OllamaModelFactory().create_model(format=None)
    response = chat_model.invoke(messages)

    return {"messages": [response]}

def create_agent_graph(tools: List[BaseTool], system_prompt: str, streaming: bool = False) -> StateGraph:
    builder = StateGraph(AgentState)
    
    # Define nodes
    builder.add_node("assistant", lambda s: assistant(s, tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("final_answer", final_answer_async if streaming else final_answer_sync)

    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "final_answer")
    builder.add_edge("final_answer", END)
    
    return builder.compile()

def run_agent_graph(
        graph, 
        user_prompt: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
    }
    final_state = graph.invoke(initial_state)
    final_response = final_state["messages"][-1]
    logger.info(f"Final response: {getattr(final_response, 'content', '[no content]')}")
    return final_response.content if hasattr(final_response, 'content') else str(final_response)

async def run_agent_graph_streaming(
        graph, 
        user_prompt: str
        ) -> AsyncGenerator[str, None]:
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
    }
    try:
        logger.debug("Running the Agent Graph Streaming Function...")
        final_state = await graph.ainvoke(initial_state) # Run full graph
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
