"""
Directorium Agent - Main Entry Point

A stateful conversational agent with persistent memory using LangGraph.
Supports multiple conversation sessions via thread_id.
Includes Human-in-the-Loop (HITL) safety with batch confirmation for write operations.
"""

import argparse
import os
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
WRITE_TOOLS = {"move_file_tool", "create_folder_tool", "rename_file_tool"}

# Confirmation keywords (case-insensitive)
CONFIRM_KEYWORDS = {"y", "yes"}
CANCEL_KEYWORDS = {"n", "no", "cancel", "abort"}


def print_banner():
    """Print the Directorium welcome banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                    DIRECTORIUM AGENT                         ║
║              Stateful Conversational Assistant               ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
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
  - Review the pending actions, then type 'y' or 'yes' to execute ALL
  - Type 'n' or 'no' to cancel all pending actions
  - Any other input clears pending actions and processes as new request

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


def parse_pending_action(pending_str: str) -> dict:
    """
    Parse a PENDING_ACTION string into a structured dict.

    Format: "PENDING_ACTION: tool_name | param1='value1' | param2='value2'"

    Args:
        pending_str: The PENDING_ACTION string from a tool

    Returns:
        Dict with 'tool_name' and 'args' keys, or None if parsing fails
    """
    if not pending_str.startswith("PENDING_ACTION:"):
        return None

    try:
        # Remove prefix and split by |
        content = pending_str[len("PENDING_ACTION:"):].strip()
        parts = [p.strip() for p in content.split("|")]

        if not parts:
            return None

        tool_name = parts[0]
        args = {}

        # Parse key=value pairs
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                # Skip metadata fields (like 'note', 'item_type', 'rename')
                if key not in ("note", "item_type", "rename"):
                    args[key] = value

        return {"tool_name": tool_name, "args": args}

    except Exception:
        return None


def execute_pending_actions(pending_queue: list) -> list:
    """
    Execute all pending actions with confirmed=True.

    Args:
        pending_queue: List of pending action dicts

    Returns:
        List of result strings from each execution
    """
    # Map tool names to functions
    tool_map = {
        "move_file": move_file,
        "create_folder": create_folder,
        "rename_file": rename_file,
    }

    results = []

    for action in pending_queue:
        tool_name = action.get("tool_name")
        args = action.get("args", {})

        if tool_name not in tool_map:
            results.append(f"Error: Unknown tool '{tool_name}'")
            continue

        func = tool_map[tool_name]

        try:
            # Execute with confirmed=True
            result = func(**args, confirmed=True)
            results.append(result)
        except Exception as e:
            results.append(f"Error executing {tool_name}: {str(e)}")

    return results


def format_pending_queue(pending_queue: list) -> str:
    """Format the pending queue for display."""
    if not pending_queue:
        return "No pending actions."

    lines = [f"{Colors.YELLOW}Pending Actions ({len(pending_queue)}):{Colors.RESET}"]

    for i, action in enumerate(pending_queue, 1):
        tool_name = action.get("tool_name", "unknown")
        args = action.get("args", {})

        if tool_name == "move_file":
            desc = f"Move: {args.get('source_path', '?')} -> {args.get('destination_path', '?')}"
        elif tool_name == "create_folder":
            desc = f"Create folder: {args.get('folder_path', '?')}"
        elif tool_name == "rename_file":
            desc = f"Rename: {args.get('old_path', '?')} -> {args.get('new_path', '?')}"
        else:
            desc = f"{tool_name}: {args}"

        lines.append(f"  {i}. {desc}")

    return "\n".join(lines)


def run_interactive_session(
    graph,
    thread_id: str,
    verbose: bool = False,
    stream: bool = True
):
    """
    Run an interactive conversation session with HITL batch support.

    Args:
        graph: The compiled LangGraph agent.
        thread_id: The conversation thread ID.
        verbose: Whether to show detailed output.
        stream: Whether to stream responses (shows tool calls in progress).
    """
    current_path = None
    pending_queue = []  # Queue of pending write operations

    print(f"{Colors.DIM}Session ID: {thread_id}{Colors.RESET}")
    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")

    while True:
        try:
            # Show pending action count in prompt if any
            if pending_queue:
                prompt = (
                    f"{Colors.YELLOW}[{len(pending_queue)} pending]{Colors.RESET} "
                    f"{Colors.GREEN}You (y/n):{Colors.RESET} "
                )
            else:
                prompt = f"{Colors.GREEN}You:{Colors.RESET} "

            # Get user input
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
                    # Generate new thread ID and clear pending queue
                    thread_id = str(uuid.uuid4())[:8]
                    pending_queue = []
                    print(f"\n{Colors.YELLOW}Started new session: {thread_id}{Colors.RESET}\n")
                    continue

                elif cmd == "/session":
                    print(f"\n{Colors.YELLOW}Current session: {thread_id}{Colors.RESET}")
                    if pending_queue:
                        print(format_pending_queue(pending_queue))
                    print()
                    continue

                elif cmd == "/pending":
                    print(f"\n{format_pending_queue(pending_queue)}\n")
                    continue

                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    print_banner()
                    continue

                else:
                    print(f"{Colors.RED}Unknown command: {user_input}{Colors.RESET}")
                    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")
                    continue

            # Handle pending queue confirmation/cancellation
            if pending_queue:
                if is_confirmation(user_input):
                    # Execute all pending actions
                    print(f"\n{Colors.BLUE}Agent:{Colors.RESET} Executing {len(pending_queue)} action(s)...\n")

                    results = execute_pending_actions(pending_queue)

                    for result in results:
                        if result.startswith("Error"):
                            print(f"  {Colors.RED}{result}{Colors.RESET}")
                        else:
                            print(f"  {Colors.GREEN}{result}{Colors.RESET}")

                    pending_queue = []
                    print()
                    continue

                elif is_cancellation(user_input):
                    count = len(pending_queue)
                    pending_queue = []
                    print(f"\n{Colors.BLUE}Agent:{Colors.RESET} {Colors.YELLOW}Cancelled {count} pending action(s).{Colors.RESET}\n")
                    continue

                else:
                    # Any other input clears pending and treats as new request
                    pending_queue = []
                    # Fall through to process as new input

            # Process the message through the agent
            print(f"\n{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)

            if stream:
                # Stream mode - show tool calls as they happen
                final_response = None
                new_pending_actions = []

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
                            # Tool results - check for PENDING_ACTION
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                content = msg.content if hasattr(msg, "content") else str(msg)

                                # Check if this is a PENDING_ACTION
                                if content.startswith("PENDING_ACTION:"):
                                    parsed = parse_pending_action(content)
                                    if parsed:
                                        new_pending_actions.append(parsed)

                                if verbose:
                                    # Truncate long tool outputs
                                    display_content = content
                                    if len(display_content) > 300:
                                        display_content = display_content[:300] + "..."
                                    print(f"{Colors.DIM}  -> {display_content}{Colors.RESET}")

                # Print final response
                if final_response:
                    print(f"\n{final_response}")

                # Add new pending actions to queue
                if new_pending_actions:
                    pending_queue.extend(new_pending_actions)
                    print(f"\n{Colors.YELLOW}{'─' * 60}{Colors.RESET}")
                    print(format_pending_queue(pending_queue))
                    print(f"{Colors.YELLOW}{'─' * 60}{Colors.RESET}")
                    print(f"{Colors.BOLD}Execute all pending actions? (y/n){Colors.RESET}")

                print()

            else:
                # Non-streaming mode
                response = invoke_agent(graph, user_input, thread_id, current_path)

                # Check if response contains PENDING_ACTION markers
                if "PENDING_ACTION:" in response:
                    # Parse any pending actions from the response
                    for line in response.split("\n"):
                        if line.strip().startswith("PENDING_ACTION:"):
                            parsed = parse_pending_action(line.strip())
                            if parsed:
                                pending_queue.append(parsed)

                print(f"{response}")

                if pending_queue:
                    print(f"\n{Colors.YELLOW}{'─' * 60}{Colors.RESET}")
                    print(format_pending_queue(pending_queue))
                    print(f"{Colors.YELLOW}{'─' * 60}{Colors.RESET}")
                    print(f"{Colors.BOLD}Execute all pending actions? (y/n){Colors.RESET}")

                print()

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Use /quit to exit{Colors.RESET}\n")
            pending_queue = []  # Clear pending on interrupt
            continue

        except EOFError:
            print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}")
            break

        except Exception as e:
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}\n")
            pending_queue = []  # Clear pending on error
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
                pending_actions = []

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
                                if content.startswith("PENDING_ACTION:"):
                                    parsed = parse_pending_action(content)
                                    if parsed:
                                        pending_actions.append(parsed)
                                if args.verbose and len(content) > 200:
                                    content = content[:200] + "..."
                                if args.verbose:
                                    print(f"{Colors.DIM}  -> {content}{Colors.RESET}")

                if final_response:
                    print(f"\n{final_response}")

                # Show pending actions for single query mode
                if pending_actions:
                    print(f"\n{Colors.YELLOW}Note: {len(pending_actions)} action(s) staged but not executed in single-query mode.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Use interactive mode to confirm and execute.{Colors.RESET}")
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
