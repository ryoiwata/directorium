"""
Tool for moving files with whitelist-based security.
"""

import shutil

from agent_core.tools.path_security import is_path_authorized


move_file_schema = {
    "type": "function",
    "function": {
        "name": "move_file",
        "description": (
            "Moves a file or directory from source to destination. Both paths "
            "must be absolute and within authorized directories defined in the "
            "whitelist. Can also be used to move and rename in one operation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the file or directory to move. "
                        "Must be within an authorized directory."
                    ),
                },
                "destination_path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the destination. Must be within "
                        "an authorized directory. If destination is a directory, "
                        "the source will be moved into it."
                    ),
                },
            },
            "required": ["source_path", "destination_path"],
        },
    },
}


def move_file(source_path, destination_path, **kwargs):
    """
    Move a file or directory from source to destination.

    Both paths are validated against the whitelist before any operation.

    Args:
        source_path: The absolute path to the file/directory to move.
        destination_path: The absolute path to the destination.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A success message or an error message prefixed with "Error:"
    """
    try:
        # SECURITY: Validate source path against whitelist
        is_authorized, resolved_source, error = is_path_authorized(source_path)
        if not is_authorized:
            return error

        # SECURITY: Validate destination path against whitelist
        is_authorized, resolved_dest, error = is_path_authorized(destination_path)
        if not is_authorized:
            return error

        # Check if source exists
        import os
        if not os.path.exists(resolved_source):
            return f'Error: Source path does not exist: "{source_path}"'

        # Perform the move operation
        shutil.move(resolved_source, resolved_dest)

        return (
            f'Successfully moved "{source_path}" to "{destination_path}"'
        )

    except PermissionError:
        return f'Error: Permission denied moving "{source_path}"'
    except shutil.Error as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
