"""
LangGraph-based stateful conversational agent for Directorium.

Defines the agent graph with:
- StateGraph for state management
- SqliteSaver for conversation persistence
- Conditional routing between agent and tool nodes
"""

import os
import sys
from typing import Annotated, TypedDict, Sequence, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.normpath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agent_core.tools.get_files_info import get_files_info  # noqa: E402
from agent_core.tools.get_file_content import get_file_content  # noqa: E402
from agent_core.providers.prompt_loader import (  # noqa: E402
    get_active_system_prompt,
)


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------

class DirectoriumState(TypedDict):
    """
    State schema for the Directorium agent.

    Attributes:
        messages: Conversation history with automatic message accumulation.
            Uses add_messages reducer to properly handle message updates.
        current_path: The current working path context for the agent.
            Can be used by tools or prompts to maintain path awareness.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_path: Optional[str]


# -----------------------------------------------------------------------------
# Tool Definitions (LangChain-compatible)
# -----------------------------------------------------------------------------

from langchain_core.tools import tool


@tool
def get_files_info_tool(path: str) -> str:
    """
    Lists files in a specified directory.

    Requires an absolute path that must be within one of the authorized
    directories defined in the whitelist. Provides file size and directory
    status for each item.

    Args:
        path: The absolute path to the directory to list. Must be within
              an authorized directory from the whitelist.
    """
    return get_files_info(path=path)


@tool
def get_file_content_tool(path: str) -> str:
    """
    Reads the content of a file.

    Requires an absolute path that must be within one of the authorized
    directories defined in the whitelist.

    Args:
        path: The absolute path to the file to read. Must be within
              an authorized directory from the whitelist.
    """
    return get_file_content(path=path)


# List of tools for the agent
tools = [get_files_info_tool, get_file_content_tool]


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------

def create_agent_graph(
    model_name: str = None,
    temperature: float = 0,
    db_path: str = "directorium_memory.db"
) -> tuple:
    """
    Create the Directorium agent graph with persistence.

    Args:
        model_name: OpenAI model name. Defaults to OPENAI_MODEL env var.
        temperature: Model temperature setting. Defaults to 0.
        db_path: Path to SQLite database for conversation persistence.

    Returns:
        tuple: (compiled_graph, checkpointer) - The compiled graph and its
               SqliteSaver checkpointer for session management.
    """
    # Get model from environment if not specified
    if model_name is None:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Initialize LLM with tools
    llm_kwargs = {"model": model_name}
    if temperature != 0:
        llm_kwargs["temperature"] = temperature

    llm = ChatOpenAI(**llm_kwargs)
    llm_with_tools = llm.bind_tools(tools)

    # Load system prompt
    system_template, parameters = get_active_system_prompt()

    # -------------------------------------------------------------------------
    # Node Functions
    # -------------------------------------------------------------------------

    def agent_node(state: DirectoriumState) -> dict:
        """
        The agent node - invokes the LLM with current conversation state.

        Injects the system message if not present, then calls the model.
        """
        messages = list(state["messages"])

        # Inject system message if not present at the start
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_template)] + messages

        # Invoke the model
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    # -------------------------------------------------------------------------
    # Build the Graph
    # -------------------------------------------------------------------------

    # Create the state graph
    graph_builder = StateGraph(DirectoriumState)

    # Add nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    # Set entry point
    graph_builder.set_entry_point("agent")

    # Add conditional edge from agent
    # tools_condition returns "tools" if there are tool calls, otherwise END
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    # Add edge from tools back to agent
    graph_builder.add_edge("tools", "agent")

    # Create checkpointer for persistence
    checkpointer = SqliteSaver.from_conn_string(db_path)

    # Compile the graph with the checkpointer
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph, checkpointer


def get_thread_config(thread_id: str) -> dict:
    """
    Create a configuration dict for a specific conversation thread.

    Args:
        thread_id: Unique identifier for the conversation session.

    Returns:
        dict: Configuration dict to pass to graph.invoke() or graph.stream()
    """
    return {"configurable": {"thread_id": thread_id}}


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def invoke_agent(
    graph,
    user_message: str,
    thread_id: str,
    current_path: Optional[str] = None
) -> str:
    """
    Invoke the agent with a user message and return the response.

    Args:
        graph: The compiled LangGraph agent.
        user_message: The user's input message.
        thread_id: The conversation thread ID for persistence.
        current_path: Optional current working path context.

    Returns:
        str: The agent's response content.
    """
    from langchain_core.messages import HumanMessage

    # Prepare input state
    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "current_path": current_path,
    }

    # Get thread config
    config = get_thread_config(thread_id)

    # Invoke the graph
    result = graph.invoke(input_state, config)

    # Extract the last AI message
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content
        return str(last_message)

    return "No response generated."


def stream_agent(
    graph,
    user_message: str,
    thread_id: str,
    current_path: Optional[str] = None
):
    """
    Stream the agent's response for a user message.

    Yields intermediate states as the agent processes, useful for
    showing tool calls and intermediate responses.

    Args:
        graph: The compiled LangGraph agent.
        user_message: The user's input message.
        thread_id: The conversation thread ID for persistence.
        current_path: Optional current working path context.

    Yields:
        dict: State updates from each node in the graph.
    """
    from langchain_core.messages import HumanMessage

    # Prepare input state
    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "current_path": current_path,
    }

    # Get thread config
    config = get_thread_config(thread_id)

    # Stream the graph execution
    for event in graph.stream(input_state, config, stream_mode="updates"):
        yield event
