"""
Tool for renaming files with whitelist-based security and confirmation requirement.
"""

import os

from agent_core.tools.path_security import is_path_authorized


rename_file_schema = {
    "type": "function",
    "function": {
        "name": "rename_file",
        "description": (
            "Renames a file or directory. Both old and new paths must be "
            "absolute and within authorized directories. Typically used for "
            "renaming within the same directory. "
            "STAGING MODE: Always call with confirmed=false first. Only set "
            "confirmed=true after user explicitly approves with 'y' or 'yes'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the existing file or directory "
                        "to rename. Must be within an authorized directory."
                    ),
                },
                "new_path": {
                    "type": "string",
                    "description": (
                        "The absolute path for the new name. Must be within "
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
            "required": ["old_path", "new_path"],
        },
    },
}


def rename_file(old_path, new_path, confirmed=False, **kwargs):
    """
    Rename a file or directory.

    Both paths are validated against the whitelist during both staging
    and execution phases for maximum security.

    Args:
        old_path: The absolute path to the file/directory to rename.
        new_path: The absolute path for the new name.
        confirmed: If False, return PENDING_ACTION. If True, execute the rename.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        PENDING_ACTION string (if not confirmed), success message, or error.
    """
    try:
        # SECURITY: Validate old path against whitelist (both phases)
        is_authorized, resolved_old, error = is_path_authorized(old_path)
        if not is_authorized:
            return error

        # SECURITY: Validate new path against whitelist (both phases)
        is_authorized, resolved_new, error = is_path_authorized(new_path)
        if not is_authorized:
            return error

        # Check if source exists
        if not os.path.exists(resolved_old):
            return f'Error: Path does not exist: "{old_path}"'

        # Check if destination already exists
        if os.path.exists(resolved_new):
            return (
                f'Error: Destination already exists: "{new_path}". '
                f'Use move_file to overwrite.'
            )

        # Determine what type of item we're renaming
        item_type = "directory" if os.path.isdir(resolved_old) else "file"
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)

        # STAGING: Return PENDING_ACTION if not confirmed
        if not confirmed:
            return (
                f"PENDING_ACTION: rename_file | "
                f"old_path='{old_path}' | "
                f"new_path='{new_path}' | "
                f"rename='{old_name}' -> '{new_name}' | "
                f"item_type={item_type}"
            )

        # EXECUTION: Perform the rename operation
        os.rename(resolved_old, resolved_new)

        return f'Successfully renamed {item_type} "{old_name}" to "{new_name}"'

    except PermissionError:
        return f'Error: Permission denied renaming "{old_path}"'
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
