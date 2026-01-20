import argparse
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.normpath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agent_core.providers.prompt_loader import (  # noqa: E402
    get_active_system_prompt
)
from agent_core.call_function import (  # noqa: E402
    available_tools,
    call_function,
)
from langchain_core.messages import (  # noqa: E402
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def main():
    parser = argparse.ArgumentParser(
        description="Query OpenAI API with a custom prompt"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=(
            """What are often overlooked tips for AI engineering?
            Use one paragraph maximum."""
        ),
        help="The query/prompt to send to OpenAI API"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information including prompt and token usage"
    )
    args = parser.parse_args()

    load_dotenv()

    prompt = args.query
    model = os.environ.get("OPENAI_MODEL")

    # Load active system prompt and parameters from YAML
    system_template, parameters = get_active_system_prompt()
    temperature = parameters.get("temperature", 0)

    # Initialize LangChain ChatOpenAI model with parameters from YAML
    # Some models don't support temperature=0, so we'll use None (default) if 0
    llm_kwargs = {"model": model}
    if temperature != 0:
        llm_kwargs["temperature"] = temperature
    llm = ChatOpenAI(**llm_kwargs)

    # Bind tools to the model with auto tool selection
    llm_with_tools = llm.bind_tools(available_tools, tool_choice="auto")

    # Initialize conversation history with system message and user prompt
    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=prompt),
    ]

    # Print initial messages if verbose
    if args.verbose:
        print("=" * 80)
        print("Initial messages:")
        print("=" * 80)
        for i, msg in enumerate(messages, 1):
            print(f"\nMessage {i}:")
            print(f"  Type: {msg.type}")
            content_preview = (
                f"{msg.content[:200]}..."
                if len(str(msg.content)) > 200
                else str(msg.content)
            )
            print(f"  Content: {content_preview}")
        print("=" * 80)
        print()

    # Agent loop with max 20 iterations
    # Note: 20 iterations supports multi-step file search operations:
    # - Typical 4-level deep directory crawl: ~5-6 iterations
    # - Broad search across multiple directories: ~10-15 iterations
    # - Increase if working with deeply nested project structures
    MAX_ITERATIONS = 20
    response_content = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(MAX_ITERATIONS):
        if args.verbose:
            print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")

        # Invoke the model with current messages
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            # If temperature=0 is not supported, retry with default temp
            if "temperature" in str(e).lower() and temperature == 0:
                llm = ChatOpenAI(model=model)
                llm_with_tools = llm.bind_tools(
                    available_tools, tool_choice="auto"
                )
                response = llm_with_tools.invoke(messages)
            else:
                raise

        # Capture model response: append to messages list
        messages.append(response)

        # Extract token usage and accumulate
        if (hasattr(response, "response_metadata") and
                response.response_metadata):
            usage = response.response_metadata.get("token_usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

        # Handle tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                # Call the function and get the result
                result_dict = call_function(tool_call, verbose=args.verbose)

                # Extract tool_call_id for ToolMessage
                if isinstance(tool_call, dict):
                    tool_call_id = (
                        tool_call.get("id") or
                        tool_call.get("tool_call_id") or
                        f"call_{id(tool_call)}"
                    )
                else:
                    tool_call_id = getattr(
                        tool_call, "id",
                        getattr(
                            tool_call, "tool_call_id", f"call_{id(tool_call)}"
                        )
                    )

                # Create ToolMessage and add to message history
                tool_message = ToolMessage(
                    content=result_dict["content"],
                    tool_call_id=tool_call_id
                )
                messages.append(tool_message)

                # Print result if verbose
                if args.verbose:
                    print(f"-> {result_dict['content']}")

            # Continue loop to process tool results
            continue
        else:
            # Final text answer (no tool calls) - extract content and break
            response_content = response.content
            if args.verbose:
                print(f"\nFinal response: {response_content}")
            break

    # Check if we reached max iterations without a final answer
    if response_content is None:
        print(
            "Error: Maximum iterations reached without a final answer.",
            file=sys.stderr
        )
        sys.exit(1)

    # Use accumulated token usage
    prompt_tokens = total_prompt_tokens if total_prompt_tokens > 0 else None
    completion_tokens = (
        total_completion_tokens if total_completion_tokens > 0 else None
    )

    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Generate timestamped log filename (JSON format)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"session_{timestamp_str}.json"
    log_file = os.path.join(logs_dir, log_filename)

    # Prepare log entry with all required fields
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "system_prompt": system_template,
        "prompt": prompt,
        "response": response_content,
        "usage": {
            "prompt_tokens": (
                prompt_tokens if prompt_tokens is not None else None
            ),
            "completion_tokens": (
                completion_tokens if completion_tokens is not None else None
            )
        }
    }

    # Write pretty-printed JSON log entry to file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=4, ensure_ascii=False)

    # Handle verbose output
    if args.verbose:
        # Show truncated system prompt
        system_preview = (
            system_template[:100] + "..."
            if len(system_template) > 100
            else system_template
        )
        print(f"System Prompt (truncated): {system_preview}")
        print(f"User prompt: {prompt}")
        if prompt_tokens is not None and completion_tokens is not None:
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Response tokens: {completion_tokens}")
        else:
            print("Prompt tokens: N/A")
            print("Response tokens: N/A")
        print()
        # Only print response_content if it exists (not None for tool calls)
        if response_content is not None:
            print(response_content)
    else:
        # Only print response_content if it exists (not None for tool calls)
        if response_content is not None:
            print(response_content)


if __name__ == "__main__":
    main()
