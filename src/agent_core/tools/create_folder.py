"""
Tool for creating folders with whitelist-based security and confirmation requirement.
"""

import os

from agent_core.tools.path_security import is_path_authorized


create_folder_schema = {
    "type": "function",
    "function": {
        "name": "create_folder",
        "description": (
            "Creates a new folder (directory) at the specified path. The path "
            "must be absolute and within an authorized directory. Creates parent "
            "directories as needed (like mkdir -p). "
            "STAGING MODE: Always call with confirmed=false first. Only set "
            "confirmed=true after user explicitly approves with 'y' or 'yes'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": (
                        "The absolute path where the folder should be created. "
                        "Must be within an authorized directory."
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
            "required": ["folder_path"],
        },
    },
}


def create_folder(folder_path, confirmed=False, **kwargs):
    """
    Create a folder at the specified path.

    The path is validated against the whitelist during both staging
    and execution phases for maximum security.

    Args:
        folder_path: The absolute path where the folder should be created.
        confirmed: If False, return PENDING_ACTION. If True, create the folder.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        PENDING_ACTION string (if not confirmed), success message, or error.
    """
    try:
        # SECURITY: Validate path against whitelist (both phases)
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

        # Determine if parent directories need to be created
        parent_dir = os.path.dirname(resolved_path)
        will_create_parents = not os.path.exists(parent_dir)

        # STAGING: Return PENDING_ACTION if not confirmed
        if not confirmed:
            if will_create_parents:
                return (
                    f"PENDING_ACTION: create_folder | "
                    f"folder_path='{folder_path}' | "
                    f"note=will create parent directories"
                )
            else:
                return (
                    f"PENDING_ACTION: create_folder | "
                    f"folder_path='{folder_path}'"
                )

        # EXECUTION: Create the folder (and any parent directories)
        os.makedirs(resolved_path, exist_ok=True)

        return f'Successfully created folder: "{folder_path}"'

    except PermissionError:
        return f'Error: Permission denied creating folder "{folder_path}"'
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
