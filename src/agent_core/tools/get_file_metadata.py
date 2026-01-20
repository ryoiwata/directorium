"""
Tool for retrieving file metadata with whitelist-based security.
"""

import json
import os
from datetime import datetime

from agent_core.tools.path_security import is_path_authorized


get_file_metadata_schema = {
    "type": "function",
    "function": {
        "name": "get_file_metadata",
        "description": (
            "Retrieves metadata about a file including size, extension, and "
            "last modified date. The path must be absolute and within an "
            "authorized directory defined in the whitelist. Returns a JSON "
            "string with the metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "The absolute path to the file. Must be within an "
                        "authorized directory from the whitelist."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
}


def get_file_metadata(file_path, **kwargs):
    """
    Get metadata about a file.

    The path is validated against the whitelist before any operation.

    Args:
        file_path: The absolute path to the file.
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        A JSON string containing file metadata, or an error message
        prefixed with "Error:"
    """
    try:
        # SECURITY: Validate path against whitelist
        is_authorized, resolved_path, error = is_path_authorized(file_path)
        if not is_authorized:
            return error

        # Check if path exists
        if not os.path.exists(resolved_path):
            return f'Error: Path does not exist: "{file_path}"'

        # Get file stats
        stat_info = os.stat(resolved_path)

        # Get file size
        size_bytes = stat_info.st_size

        # Get last modified time
        mtime = stat_info.st_mtime
        last_modified = datetime.fromtimestamp(mtime).isoformat()

        # Get extension (empty string for directories or files without extension)
        _, extension = os.path.splitext(resolved_path)

        # Get file name
        file_name = os.path.basename(resolved_path)

        # Determine if it's a file or directory
        is_directory = os.path.isdir(resolved_path)

        # Build metadata dictionary
        metadata = {
            "path": file_path,
            "name": file_name,
            "size_bytes": size_bytes,
            "size_human": _format_size(size_bytes),
            "extension": extension if extension else None,
            "last_modified": last_modified,
            "is_directory": is_directory,
        }

        # Add additional info for directories
        if is_directory:
            try:
                items = os.listdir(resolved_path)
                metadata["item_count"] = len(items)
            except PermissionError:
                metadata["item_count"] = None

        return json.dumps(metadata, indent=2)

    except PermissionError:
        return f'Error: Permission denied accessing "{file_path}"'
    except OSError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _format_size(size_bytes):
    """
    Format a size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
