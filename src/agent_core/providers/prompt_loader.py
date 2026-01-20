import os
import yaml


def get_settings():
    """
    Load settings from config/settings.yaml.

    Returns:
        dict: Dictionary containing all settings from the YAML file
    """
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", "..", ".."))

    # Read settings from config/settings.yaml
    config_path = os.path.join(project_root, "config", "settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    return settings


def get_active_system_prompt():
    """
    Load the active system prompt from YAML configuration.

    Returns:
        tuple: (template_string, parameters_dict)
    """
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", "..", ".."))

    # Read settings to get active prompt name
    settings = get_settings()
    active_prompt_name = settings.get("active_prompt", "v1_robot")

    # Load the corresponding prompt file
    prompt_path = os.path.join(
        project_root, "system_prompts", f"{active_prompt_name}.yaml"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    template = prompt_data.get("template", "")
    parameters = prompt_data.get("parameters", {})

    return template, parameters

