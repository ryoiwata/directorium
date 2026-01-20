"""
Tool for moving files with whitelist-based security and confirmation requirement.
"""

import os
import shutil

from agent_core.tools.path_security import is_path_authorized


move_file_schema = {
    "type": "function",
    "function": {
        "name": "move_file",
        "description": (
            "Moves a file or directory from source to destination. Both paths "
            "must be absolute and within authorized directories. "
            "STAGING MODE: Always call with confirmed=false first. Only set "
            "confirmed=true after user explicitly approves with 'y' or 'yes'."
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
                        "an authorized directory."
                    ),
                },
                "confirmed": {
                    "type": "boolean",
                    "description": (
                        "Set to false to stage the action (returns PENDING_ACTION). "
                        "Set to true only after user confirms with 'y' or 'yes'."
                    ),
                    "default": False,
                },
            },
            "required": ["source_path", "destination_path"],
        },
    },
}


def move_file(source_path, destination_path, confirmed=False, **kwargs):
    """
    Move a file or directory from source to destination.

    Both paths are validated against the whitelist during both staging
    and execution phases for maximum security.

    Args:
        source_path: The absolute path to the file/directory to move.
        destination_path: The absolute path to the destination.
        confirmed: If False, return PENDING_ACTION. If True, execute the move.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        PENDING_ACTION string (if not confirmed), success message, or error.
    """
    try:
        # SECURITY: Validate source path against whitelist (both phases)
        is_authorized, resolved_source, error = is_path_authorized(source_path)
        if not is_authorized:
            return error

        # SECURITY: Validate destination path against whitelist (both phases)
        is_authorized, resolved_dest, error = is_path_authorized(destination_path)
        if not is_authorized:
            return error

        # Check if source exists
        if not os.path.exists(resolved_source):
            return f'Error: Source path does not exist: "{source_path}"'

        # Determine what type of item we're moving
        item_type = "directory" if os.path.isdir(resolved_source) else "file"

        # STAGING: Return PENDING_ACTION if not confirmed
        if not confirmed:
            return (
                f"PENDING_ACTION: move_file | "
                f"source_path='{source_path}' | "
                f"destination_path='{destination_path}' | "
                f"item_type={item_type}"
            )

        # EXECUTION: Perform the move operation
        shutil.move(resolved_source, resolved_dest)

        return (
            f'Successfully moved {item_type} "{source_path}" to "{destination_path}"'
        )

    except PermissionError:
        return f'Error: Permission denied moving "{source_path}"'
    except shutil.Error as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
