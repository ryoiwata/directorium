"""
Tool for creating folders with whitelist-based security.
"""

import os

from agent_core.tools.path_security import is_path_authorized


create_folder_schema = {
    "type": "function",
    "function": {
        "name": "create_folder",
        "description": (
            "Creates a new folder (directory) at the specified path. The path "
            "must be absolute and within an authorized directory defined in the "
            "whitelist. Creates parent directories as needed (like mkdir -p)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": (
                        "The absolute path where the folder should be created. "
                        "Must be within an authorized directory. Parent "
                        "directories will be created if they don't exist."
                    ),
                },
            },
            "required": ["folder_path"],
        },
    },
}


def create_folder(folder_path, **kwargs):
    """
    Create a folder at the specified path.

    The path is validated against the whitelist before any operation.
    Creates parent directories as needed (exist_ok=True behavior).

    Args:
        folder_path: The absolute path where the folder should be created.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A success message or an error message prefixed with "Error:"
    """
    try:
        # SECURITY: Validate path against whitelist
        is_authorized, resolved_path, error = is_path_authorized(folder_path)
        if not is_authorized:
            return error

        # Check if folder already exists
        if os.path.exists(resolved_path):
            if os.path.isdir(resolved_path):
                return f'Folder already exists: "{folder_path}"'
            else:
                return (
                    f'Error: A file already exists at this path: "{folder_path}"'
                )

        # Create the folder (and any parent directories)
        os.makedirs(resolved_path, exist_ok=True)

        return f'Successfully created folder: "{folder_path}"'

    except PermissionError:
        return f'Error: Permission denied creating folder "{folder_path}"'
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
