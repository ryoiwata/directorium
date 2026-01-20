"""
Tool for renaming files with whitelist-based security.
"""

import os

from agent_core.tools.path_security import is_path_authorized


rename_file_schema = {
    "type": "function",
    "function": {
        "name": "rename_file",
        "description": (
            "Renames a file or directory. Both old and new paths must be "
            "absolute and within authorized directories defined in the "
            "whitelist. Typically used for renaming within the same directory."
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
                        "an authorized directory. Should typically be in the "
                        "same directory as the original."
                    ),
                },
            },
            "required": ["old_path", "new_path"],
        },
    },
}


def rename_file(old_path, new_path, **kwargs):
    """
    Rename a file or directory.

    Both paths are validated against the whitelist before any operation.

    Args:
        old_path: The absolute path to the file/directory to rename.
        new_path: The absolute path for the new name.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A success message or an error message prefixed with "Error:"
    """
    try:
        # SECURITY: Validate old path against whitelist
        is_authorized, resolved_old, error = is_path_authorized(old_path)
        if not is_authorized:
            return error

        # SECURITY: Validate new path against whitelist
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

        # Perform the rename operation
        os.rename(resolved_old, resolved_new)

        return f'Successfully renamed "{old_path}" to "{new_path}"'

    except PermissionError:
        return f'Error: Permission denied renaming "{old_path}"'
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
