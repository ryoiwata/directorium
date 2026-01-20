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
            "MANDATORY STAGING: Set confirmed=False to propose the action for "
            "user review. The tool will return a STAGED_ACTION response. "
            "Set confirmed=True ONLY if the user has just provided explicit "
            "permission (y/yes) for this specific move operation. "
            "NEVER set confirmed=True without prior user approval."
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
                        "Set to False to propose/stage the action (returns STAGED_ACTION). "
                        "Set to True ONLY after user explicitly confirms with 'y' or 'yes' "
                        "for THIS SPECIFIC action. Default is False."
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
        confirmed: If False, return STAGED_ACTION. If True, execute the move.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        STAGED_ACTION string (if not confirmed), success message, or error.
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

        # STAGING: Return STAGED_ACTION if not confirmed
        if not confirmed:
            return (
                f"STAGED_ACTION: move_file -> "
                f"source='{source_path}', destination='{destination_path}'"
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
