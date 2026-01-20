"""
Tool for reading file contents with whitelist-based security.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.normpath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agent_core.providers.prompt_loader import get_settings  # noqa: E402
from agent_core.tools.path_security import is_path_authorized  # noqa: E402


get_file_content_schema = {
    "type": "function",
    "function": {
        "name": "get_file_content",
        "description": (
            "Reads the content of a file. Requires an absolute path that must "
            "be within one of the authorized directories defined in the whitelist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the file to read. Must be within "
                        "an authorized directory from the whitelist."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


def get_file_content(path, **kwargs):
    """
    Get the content of a file with whitelist-based security validation.

    The path is validated against the whitelist before any file access occurs.

    Args:
        path: The absolute path to the file to read.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A string with file content or an error message prefixed with "Error:"
    """
    try:
        # SECURITY: First step - validate path against whitelist
        is_authorized, resolved_path, error = is_path_authorized(path)
        if not is_authorized:
            return error

        # Load MAX_CHARS from settings
        settings = get_settings()
        MAX_CHARS = settings.get("MAX_CHARS", 10000)

        target_file = resolved_path

        # Check if target_file is a regular file
        if not os.path.isfile(target_file):
            return (
                f'Error: File not found or is not a regular file: '
                f'"{path}"'
            )

        # Read file content with MAX_CHARS limit
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read(MAX_CHARS)
            # Check if file was truncated
            remaining = f.read(1)
            if remaining:
                content += (
                    f'\n[...File "{path}" truncated at {MAX_CHARS} '
                    f'characters]'
                )

        return content

    except UnicodeDecodeError:
        return f'Error: Cannot read "{path}" - file is not valid UTF-8 text'
    except PermissionError:
        return f'Error: Permission denied reading "{path}"'
    except Exception as e:
        return f"Error: {str(e)}"
