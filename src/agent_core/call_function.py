"""
Function dispatcher for Directorium agent tools.

Routes tool calls to their implementations without enforcing any path context.
Each tool is responsible for validating paths using the central path_security module.
"""

import copy
import json

from agent_core.tools.get_files_info import (
    get_files_info,
    get_files_info_schema,
)
from agent_core.tools.get_file_content import (
    get_file_content,
    get_file_content_schema,
)

available_tools = [
    get_files_info_schema,
    get_file_content_schema,
]

# Map tool names to their function implementations
function_map = {
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
}


def call_function(tool_call, verbose=False):
    """
    Execute a tool call and return the result.

    The agent is responsible for providing the path it wants to access.
    Each tool validates the path against the whitelist before proceeding.

    Args:
        tool_call: Tool call object from LangChain (dict or ToolCall object)
        verbose: If True, print detailed function call info

    Returns:
        dict: Dictionary with 'content' key containing the result string,
              compatible with LangChain's ToolMessage format
    """
    # Extract tool name and args from tool_call
    if isinstance(tool_call, dict):
        tool_name = (
            tool_call.get("name") or
            tool_call.get("function", {}).get("name", "unknown")
        )
        tool_args = (
            tool_call.get("args") or
            tool_call.get("function", {}).get("arguments", {})
        )
    else:
        # ToolCall object - try attribute access first, then dict
        tool_name = getattr(tool_call, "name", None)
        if tool_name is None and hasattr(tool_call, "get"):
            tool_name = tool_call.get("name", "unknown")
        if tool_name is None:
            tool_name = "unknown"

        tool_args = getattr(tool_call, "args", None)
        if tool_args is None and hasattr(tool_call, "get"):
            tool_args = tool_call.get("args", {})
        if tool_args is None:
            tool_args = {}

    # If args is a JSON string, parse it
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            pass

    # Print calling function info
    if verbose:
        print(f"Calling function: {tool_name}({tool_args})")
    else:
        print(f"- Calling function: {tool_name}")

    # Get the function from function_map
    if tool_name not in function_map:
        return {
            "content": f"Error: Unknown function '{tool_name}'"
        }

    func = function_map[tool_name]

    # Create a shallow copy of args to avoid modifying the original
    args_copy = copy.copy(tool_args) if tool_args else {}
    if not isinstance(args_copy, dict):
        args_copy = {}

    # Call the function with the provided arguments
    # Each tool is responsible for its own path validation
    try:
        result = func(**args_copy)
        return {"content": result}
    except TypeError as e:
        # Handle missing or unexpected arguments
        return {"content": f"Error: Invalid arguments - {str(e)}"}
    except Exception as e:
        return {"content": f"Error: {str(e)}"}
