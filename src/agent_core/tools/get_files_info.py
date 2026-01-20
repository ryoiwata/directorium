"""
Tool for listing directory contents with whitelist-based security.
"""

import argparse
import os
from pathlib import Path

from agent_core.tools.path_security import is_path_authorized, get_whitelist


get_files_info_schema = {
    "type": "function",
    "function": {
        "name": "get_files_info",
        "description": (
            "Lists files in a specified directory. Requires an absolute path "
            "that must be within one of the authorized directories defined in "
            "the whitelist. Provides file size and directory status for each item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the directory to list. Must be "
                        "within an authorized directory from the whitelist."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


def get_files_info(path, **kwargs):
    """
    Get information about files in a directory with whitelist-based security.

    The path is validated against the whitelist before any directory access occurs.

    Args:
        path: The absolute path to the directory to list.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A string with file information or an error message prefixed with "Error:"
    """
    try:
        # SECURITY: First step - validate path against whitelist
        is_authorized, resolved_path, error = is_path_authorized(path)
        if not is_authorized:
            return error

        target_dir = resolved_path

        # Check if target_dir is a directory
        if not os.path.isdir(target_dir):
            return f'Error: "{path}" is not a directory'

        # Iterate through items in the directory
        results = []
        items = sorted(os.listdir(target_dir))

        for item in items:
            item_path = os.path.join(target_dir, item)
            try:
                size = os.path.getsize(item_path)
                is_dir = os.path.isdir(item_path)
                results.append(
                    f"- {item}: file_size={size} bytes, is_dir={is_dir}"
                )
            except (OSError, PermissionError):
                # Skip items we can't access
                continue

        # Return formatted string with each item on a new line
        return "\n".join(results) if results else "(empty directory)"

    except PermissionError:
        return f'Error: Permission denied accessing "{path}"'
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Command-line interface for get_files_info."""
    parser = argparse.ArgumentParser(
        description=(
            "Get information about files in a directory with "
            "security guardrails"
        )
    )
    parser.add_argument(
        "path",
        type=str,
        help="Absolute path to the directory to list"
    )
    args = parser.parse_args()

    result = get_files_info(path=args.path)
    print(result)


if __name__ == "__main__":
    main()
