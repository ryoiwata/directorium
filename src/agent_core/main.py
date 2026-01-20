"""
Directorium Agent - Main Entry Point

A stateful conversational agent with persistent memory using LangGraph.
Supports multiple conversation sessions via thread_id.
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
  {Colors.GREEN}/clear{Colors.RESET}     - Clear the screen
  {Colors.GREEN}/quit{Colors.RESET}      - Exit the agent (also: /exit, /q)

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


def run_interactive_session(
    graph,
    thread_id: str,
    verbose: bool = False,
    stream: bool = True
):
    """
    Run an interactive conversation session.

    Args:
        graph: The compiled LangGraph agent.
        thread_id: The conversation thread ID.
        verbose: Whether to show detailed output.
        stream: Whether to stream responses (shows tool calls in progress).
    """
    current_path = None

    print(f"{Colors.DIM}Session ID: {thread_id}{Colors.RESET}")
    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")

    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}You:{Colors.RESET} ").strip()

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

                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    print_banner()
                    continue

                else:
                    print(f"{Colors.RED}Unknown command: {user_input}{Colors.RESET}")
                    print(f"{Colors.DIM}Type /help for available commands{Colors.RESET}\n")
                    continue

            # Process the message
            print(f"\n{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)

            if stream:
                # Stream mode - show tool calls as they happen
                final_response = None

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
                                    final_response = msg.content if hasattr(msg, "content") else str(msg)

                        elif node_name == "tools":
                            # Tool results
                            if verbose:
                                messages = node_output.get("messages", [])
                                for msg in messages:
                                    content = msg.content if hasattr(msg, "content") else str(msg)
                                    # Truncate long tool outputs
                                    if len(content) > 200:
                                        content = content[:200] + "..."
                                    print(f"{Colors.DIM}  -> {content}{Colors.RESET}")

                # Print final response
                if final_response:
                    print(f"\n{final_response}")
                print()

            else:
                # Non-streaming mode
                response = invoke_agent(graph, user_input, thread_id, current_path)
                print(f"{response}\n")

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

                        elif node_name == "tools" and args.verbose:
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                print(f"{Colors.DIM}  -> {content}{Colors.RESET}")

                if final_response:
                    print(f"\n{final_response}")
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
