"""
Central path authorization utility for Directorium.

Provides whitelist-based security validation for all file system access.
"""

import os
from pathlib import Path

import yaml


# Security error message for unauthorized access attempts
ACCESS_DENIED_ERROR = (
    "Error: Access Denied. This path is outside the authorized security zones."
)

# Cache for whitelist to avoid repeated file reads
_whitelist_cache = None
_whitelist_cache_mtime = None


def _get_config_path():
    """
    Get the path to the whitelist configuration file.

    Returns:
        Path: Path object pointing to config/whitelist.yaml
    """
    # Navigate from this file to the project config directory
    # This file: src/agent_core/tools/path_security.py
    # Config: config/whitelist.yaml
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    return project_root / "config" / "whitelist.yaml"


def _load_whitelist():
    """
    Load the whitelist from config/whitelist.yaml with caching.

    The whitelist is cached and only reloaded if the file has been modified.

    Returns:
        list[str]: List of absolute paths that are authorized for access.
    """
    global _whitelist_cache, _whitelist_cache_mtime

    config_path = _get_config_path()

    # Check if we need to reload the cache
    try:
        current_mtime = config_path.stat().st_mtime
    except (OSError, FileNotFoundError):
        # Config file doesn't exist - return empty list
        return []

    if _whitelist_cache is not None and _whitelist_cache_mtime == current_mtime:
        return _whitelist_cache

    # Load and parse the whitelist
    allowed_roots = []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config and "allowed_roots" in config:
            for root in config["allowed_roots"]:
                if root:
                    # Normalize and resolve to absolute path
                    normalized = str(Path(root).expanduser().resolve())
                    if normalized not in allowed_roots:
                        allowed_roots.append(normalized)
    except (yaml.YAMLError, IOError) as e:
        print(f"Warning: Could not load whitelist.yaml: {e}")
        return []

    # Update cache
    _whitelist_cache = allowed_roots
    _whitelist_cache_mtime = current_mtime

    return allowed_roots


def is_path_authorized(target_path):
    """
    Check if a target path is within one of the whitelisted directories.

    This is the central security validation function that must be called
    by all tools before accessing any file system resource.

    Args:
        target_path: The path to validate (string or Path object).
                     Can be relative or absolute.

    Returns:
        tuple: (is_authorized: bool, resolved_path: str or None, error: str or None)
               - is_authorized: True if the path is within a whitelisted directory
               - resolved_path: The absolute, normalized path if authorized
               - error: Error message if not authorized, None otherwise
    """
    if not target_path:
        return (False, None, "Error: No path provided.")

    allowed_roots = _load_whitelist()

    if not allowed_roots:
        return (False, None, "Error: No authorized paths configured in whitelist.")

    # Convert to Path object and resolve to absolute path
    path_obj = Path(target_path).expanduser()

    # If relative path, we cannot determine authorization without context
    # The path must be absolute for security validation
    if not path_obj.is_absolute():
        return (
            False,
            None,
            "Error: Relative paths are not supported. Please provide an absolute path."
        )

    # Resolve to canonical absolute path (resolves symlinks, .., etc.)
    try:
        resolved = str(path_obj.resolve())
    except (OSError, RuntimeError) as e:
        return (False, None, f"Error: Cannot resolve path: {e}")

    # Check if the resolved path is within any whitelisted root
    for root in allowed_roots:
        root_abs = str(Path(root).resolve())
        try:
            common_path = os.path.commonpath([root_abs, resolved])
            if common_path == root_abs:
                # Path is within this whitelisted root
                return (True, resolved, None)
        except ValueError:
            # Different drives on Windows - not a match, continue checking
            continue

    return (False, None, ACCESS_DENIED_ERROR)


def get_whitelist():
    """
    Get the current whitelist of authorized directories.

    Returns:
        list[str]: List of absolute paths that are authorized for access.
    """
    return _load_whitelist().copy()


def clear_whitelist_cache():
    """
    Clear the whitelist cache, forcing a reload on next access.

    Useful for testing or when the whitelist file has been modified.
    """
    global _whitelist_cache, _whitelist_cache_mtime
    _whitelist_cache = None
    _whitelist_cache_mtime = None
