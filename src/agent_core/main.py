"""
Directorium Agent - Main Entry Point

A stateful conversational agent with persistent memory using LangGraph.
Supports multiple conversation sessions via thread_id.
Includes Human-in-the-Loop (HITL) safety with INDIVIDUAL confirmation for write operations.
"""

import argparse
import os
import re
import sys
import uuid

from dotenv import load_dotenv

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.normpath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agent_core.graph import (  # noqa: E402
    create_agent_graph,
    stream_agent,
    invoke_agent,
)

# Import tool functions for direct execution after confirmation
from agent_core.tools.move_file import move_file  # noqa: E402
from agent_core.tools.create_folder import create_folder  # noqa: E402
from agent_core.tools.rename_file import rename_file  # noqa: E402


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Write tools that require HITL confirmation
WRITE_TOOLS = {"move_file", "create_folder", "rename_file"}

# Confirmation keywords (case-insensitive)
CONFIRM_KEYWORDS = {"y", "yes"}
CANCEL_KEYWORDS = {"n", "no", "cancel", "abort"}


def print_banner():
    """Print the Directorium welcome banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}======================================================================
                        DIRECTORIUM AGENT
              Stateful Conversational Assistant
======================================================================{Colors.RESET}
"""
    print(banner)


def print_help():
    """Print available commands."""
    help_text = f"""
{Colors.YELLOW}Available Commands:{Colors.RESET}
  {Colors.GREEN}/help{Colors.RESET}      - Show this help message
  {Colors.GREEN}/new{Colors.RESET}       - Start a new conversation session
  {Colors.GREEN}/session{Colors.RESET}   - Show current session ID
  {Colors.GREEN}/pending{Colors.RESET}   - Show pending actions queue
  {Colors.GREEN}/clear{Colors.RESET}     - Clear the screen
  {Colors.GREEN}/quit{Colors.RESET}      - Exit the agent (also: /exit, /q)

{Colors.YELLOW}File Operations (STAGING MODE):{Colors.RESET}
  - All write operations (move, rename, create folder) are staged first
  - You will be asked to confirm EACH action individually: (y/n)
  - Type 'y' or 'yes' to execute that specific action
  - Type 'n' or 'no' to skip that action and proceed to the next
  - Any other input clears ALL remaining staged actions and processes as new request

{Colors.YELLOW}Tips:{Colors.RESET}
  - Provide absolute paths when referencing files/directories
  - The agent remembers context within a session
  - Use /new to start fresh without history
"""
    print(help_text)


def format_tool_call(tool_name: str, tool_args: dict) -> str:
    """Format a tool call for display."""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
    return f"{Colors.DIM}[Tool: {tool_name}({args_str})]{Colors.RESET}"


def is_confirmation(user_input: str) -> bool:
    """Check if user input is a confirmation (y/yes only)."""
    normalized = user_input.lower().strip()
    return normalized in CONFIRM_KEYWORDS


def is_cancellation(user_input: str) -> bool:
    """Check if user input is a cancellation."""
    normalized = user_input.lower().strip()
    return normalized in CANCEL_KEYWORDS


def parse_staged_action(staged_str: str) -> dict:
    """
    Parse a STAGED_ACTION string into a structured dict.

    Format: "STAGED_ACTION: tool_name -> param1='value1', param2='value2'"

    Args:
        staged_str: The STAGED_ACTION string from a tool

    Returns:
        Dict with 'tool_name' and 'args' keys, or None if parsing fails
    """
    if not staged_str.startswith("STAGED_ACTION:"):
        return None

    try:
        # Remove prefix: "STAGED_ACTION: "
        content = staged_str[len("STAGED_ACTION:"):].strip()

        # Split by " -> " to get tool name and args
        if " -> " not in content:
            return None

        tool_name, args_str = content.split(" -> ", 1)
        tool_name = tool_name.strip()
        args_str = args_str.strip()

        # Remove trailing notes in parentheses like "(will create parent directories)"
        args_str = re.sub(r'\s*\([^)]*\)\s*$', '', args_str)

        args = {}

        # Parse key='value' pairs, handling commas within values
        # Pattern: key='value' or key="value"
        pattern = r"(\w+)='([^']*)'|(\w+)=\"([^\"]*)\""
        matches = re.findall(pattern, args_str)

        for match in matches:
            if match[0]:  # Single quote match
                key, value = match[0], match[1]
            else:  # Double quote match
                key, value = match[2], match[3]
            args[key] = value

        if not args:
            # Fallback: try simple comma split for key=value without quotes
            for part in args_str.split(", "):
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    args[key] = value

        return {"tool_name": tool_name, "args": args}

    except Exception:
        return None


def execute_single_action(action: dict) -> str:
    """
    Execute a single staged action with confirmed=True.

    Args:
        action: Dict with 'tool_name' and 'args' keys

    Returns:
        Result string from the execution
    """
    # Map tool names to functions
    tool_map = {
        "move_file": move_file,
        "create_folder": create_folder,
        "rename_file": rename_file,
    }

    tool_name = action.get("tool_name")
    args = action.get("args", {})

    if tool_name not in tool_map:
        return f"Error: Unknown tool '{tool_name}'"

    func = tool_map[tool_name]

    try:
        # Execute with confirmed=True
        result = func(**args, confirmed=True)
        return result
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def format_staged_action_prompt(action: dict) -> str:
    """Format a staged action as a user-friendly prompt."""
    tool_name = action.get("tool_name", "unknown")
    args = action.get("args", {})

    if tool_name == "move_file":
        source = args.get("source", args.get("source_path", "?"))
        dest = args.get("destination", args.get("destination_path", "?"))
        return f"Move '{source}' to '{dest}'"
    elif tool_name == "create_folder":
        folder = args.get("folder_path", "?")
        return f"Create folder '{folder}'"
    elif tool_name == "rename_file":
        old = args.get("old_path", "?")
        new = args.get("new_path", "?")
        return f"Rename '{old}' to '{new}'"
    else:
        return f"{tool_name}: {args}"


def format_staging_queue(staging_queue: list) -> str:
    """Format the staging queue for display."""
    if not staging_queue:
        return "No staged actions."

    lines = [f"{Colors.YELLOW}Staged Actions ({len(staging_queue)}):{Colors.RESET}"]

    for i, action in enumerate(staging_queue, 1):
        desc = format_staged_action_prompt(action)
        lines.append(f"  {i}. {desc}")

    return "\n".join(lines)


def process_staging_queue(staging_queue: list) -> list:
    """
    Process staged actions one by one with individual user confirmation.

    Args:
        staging_queue: List of staged action dicts

    Returns:
        List of result strings from executed actions
    """
    results = []
    remaining_queue = staging_queue.copy()

    while remaining_queue:
        action = remaining_queue[0]
        action_desc = format_staged_action_prompt(action)

        # Show remaining count if more than one
        if len(remaining_queue) > 1:
            queue_info = f" [{len(remaining_queue)} remaining]"
        else:
            queue_info = ""

        # Prompt for this specific action
        print(f"\n{Colors.YELLOW}I am ready to:{Colors.RESET} {action_desc}{queue_info}")
        prompt = f"{Colors.BOLD}Proceed? (y/n):{Colors.RESET} "

        try:
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.RED}Cancelled all remaining actions.{Colors.RESET}")
            break

        if is_confirmation(user_input):
            # Execute this action
            result = execute_single_action(action)
            if result.startswith("Error"):
                print(f"  {Colors.RED}{result}{Colors.RESET}")
            else:
                print(f"  {Colors.GREEN}{result}{Colors.RESET}")
            results.append(result)
            remaining_queue.pop(0)

        elif is_cancellation(user_input):
            # Skip this action, move to next
            print(f"  {Colors.DIM}Skipped.{Colors.RESET}")
            remaining_queue.pop(0)

        else:
            # Any other input: clear remaining queue and treat as new query
            print(f"\n{Colors.YELLOW}Clearing {len(remaining_queue)} remaining staged action(s).{Colors.RESET}")
            # Return a special marker to indicate new query
            return {"results": results, "new_query": user_input}

    return {"results": results, "new_query": None}


def run_interactive_session(
    graph,
    thread_id: str,
    verbose: bool = False,
    stream: bool = True
):
    """
    Run an interactive conversation session with INDIVIDUAL HITL confirmation.

    Args:
        graph: The compiled LangGraph agent.
        thread_id: The conversation thread ID.
        verbose: Whether to show detailed output.
        stream: Whether to stream responses (shows tool calls in progress).
    """
    current_path = None
    pending_new_query = None  # Holds a new query if user interrupts staging

    print(f"{Colors.DIM}Session ID: {thread_id}{Colors.RESET}")
    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")

    while True:
        try:
            # Check if we have a pending new query from interrupted staging
            if pending_new_query:
                user_input = pending_new_query
                pending_new_query = None
            else:
                prompt = f"{Colors.GREEN}You:{Colors.RESET} "
                user_input = input(prompt).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd in ["/quit", "/exit", "/q"]:
                    print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}")
                    break

                elif cmd == "/help":
                    print_help()
                    continue

                elif cmd == "/new":
                    # Generate new thread ID
                    thread_id = str(uuid.uuid4())[:8]
                    print(f"\n{Colors.YELLOW}Started new session: {thread_id}{Colors.RESET}\n")
                    continue

                elif cmd == "/session":
                    print(f"\n{Colors.YELLOW}Current session: {thread_id}{Colors.RESET}\n")
                    continue

                elif cmd == "/pending":
                    print(f"\n{Colors.DIM}No pending actions (staging queue is processed immediately).{Colors.RESET}\n")
                    continue

                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    print_banner()
                    continue

                else:
                    print(f"{Colors.RED}Unknown command: {user_input}{Colors.RESET}")
                    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")
                    continue

            # Process the message through the agent
            print(f"\n{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)

            if stream:
                # Stream mode - show tool calls as they happen
                final_response = None
                staged_actions = []

                for event in stream_agent(graph, user_input, thread_id, current_path):
                    # Process each event from the stream
                    for node_name, node_output in event.items():
                        if node_name == "agent":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                # Check for tool calls
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    if verbose:
                                        print()  # Newline before tool calls
                                        for tc in msg.tool_calls:
                                            print(format_tool_call(tc["name"], tc["args"]))
                                    else:
                                        print(f"{Colors.DIM}[Using tools...]{Colors.RESET}", end=" ", flush=True)
                                else:
                                    # Final response
                                    content = msg.content if hasattr(msg, "content") else str(msg)
                                    final_response = content

                        elif node_name == "tools":
                            # Tool results - check for STAGED_ACTION
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                content = msg.content if hasattr(msg, "content") else str(msg)

                                # Check if this is a STAGED_ACTION
                                if content.startswith("STAGED_ACTION:"):
                                    parsed = parse_staged_action(content)
                                    if parsed:
                                        staged_actions.append(parsed)

                                if verbose:
                                    # Truncate long tool outputs
                                    display_content = content
                                    if len(display_content) > 300:
                                        display_content = display_content[:300] + "..."
                                    print(f"{Colors.DIM}  -> {display_content}{Colors.RESET}")

                # Print final response
                if final_response:
                    print(f"\n{final_response}")

                # Process staged actions one by one with individual confirmation
                if staged_actions:
                    print(f"\n{Colors.YELLOW}{'=' * 60}{Colors.RESET}")
                    print(format_staging_queue(staged_actions))
                    print(f"{Colors.YELLOW}{'=' * 60}{Colors.RESET}")

                    result = process_staging_queue(staged_actions)

                    if isinstance(result, dict) and result.get("new_query"):
                        # User provided a new query during staging
                        pending_new_query = result["new_query"]

                print()

            else:
                # Non-streaming mode
                response = invoke_agent(graph, user_input, thread_id, current_path)

                # Check if response contains STAGED_ACTION markers
                staged_actions = []
                if "STAGED_ACTION:" in response:
                    # Parse any staged actions from the response
                    for line in response.split("\n"):
                        if line.strip().startswith("STAGED_ACTION:"):
                            parsed = parse_staged_action(line.strip())
                            if parsed:
                                staged_actions.append(parsed)

                print(f"{response}")

                if staged_actions:
                    print(f"\n{Colors.YELLOW}{'=' * 60}{Colors.RESET}")
                    print(format_staging_queue(staged_actions))
                    print(f"{Colors.YELLOW}{'=' * 60}{Colors.RESET}")

                    result = process_staging_queue(staged_actions)

                    if isinstance(result, dict) and result.get("new_query"):
                        # User provided a new query during staging
                        pending_new_query = result["new_query"]

                print()

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Use /quit to exit{Colors.RESET}\n")
            continue

        except EOFError:
            print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}")
            break

        except Exception as e:
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}\n")
            if verbose:
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for the Directorium agent."""
    parser = argparse.ArgumentParser(
        description="Directorium - A stateful conversational file system agent"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to execute (non-interactive mode)"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Conversation thread ID (generates new if not provided)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="directorium_memory.db",
        help="Path to SQLite database for conversation persistence"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including tool calls and results"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode (wait for complete response)"
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Generate or use provided thread ID
    thread_id = args.thread_id or str(uuid.uuid4())[:8]

    # Create the agent graph
    try:
        graph, checkpointer = create_agent_graph(db_path=args.db_path)
    except Exception as e:
        print(f"{Colors.RED}Failed to initialize agent: {e}{Colors.RESET}")
        sys.exit(1)

    # Single query mode
    if args.query:
        try:
            if not args.no_stream:
                # Stream mode for single query
                print(f"{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)
                final_response = None
                staged_actions = []

                for event in stream_agent(graph, args.query, thread_id):
                    for node_name, node_output in event.items():
                        if node_name == "agent":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    if args.verbose:
                                        print()
                                        for tc in msg.tool_calls:
                                            print(format_tool_call(tc["name"], tc["args"]))
                                else:
                                    final_response = msg.content if hasattr(msg, "content") else str(msg)

                        elif node_name == "tools":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                if content.startswith("STAGED_ACTION:"):
                                    parsed = parse_staged_action(content)
                                    if parsed:
                                        staged_actions.append(parsed)
                                if args.verbose and len(content) > 200:
                                    content = content[:200] + "..."
                                if args.verbose:
                                    print(f"{Colors.DIM}  -> {content}{Colors.RESET}")

                if final_response:
                    print(f"\n{final_response}")

                # Show staged actions for single query mode (no confirmation in non-interactive)
                if staged_actions:
                    print(f"\n{Colors.YELLOW}Note: {len(staged_actions)} action(s) staged but not executed in single-query mode.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Use interactive mode to confirm and execute.{Colors.RESET}")
                    for action in staged_actions:
                        print(f"  - {format_staged_action_prompt(action)}")
            else:
                response = invoke_agent(graph, args.query, thread_id)
                print(response)

        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            sys.exit(1)

        return

    # Interactive mode
    print_banner()
    run_interactive_session(
        graph,
        thread_id,
        verbose=args.verbose,
        stream=not args.no_stream
    )


if __name__ == "__main__":
    main()
