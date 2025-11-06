"""
Agent validation utilities for self-extensibility system.

Provides validation functions for generated agent configurations to ensure:
- Valid YAML syntax
- No delegation cycles (DAG structure)
- Referenced behaviors exist
- Config follows Jetbox conventions
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def validate_yaml_syntax(yaml_str: str) -> dict[str, Any]:
    """
    Validate YAML syntax.

    Args:
        yaml_str: YAML content as string

    Returns:
        dict with keys:
        - valid (bool): True if syntax is valid
        - error (str): Error message if invalid (optional)
        - data (dict): Parsed YAML data if valid (optional)
    """
    if not YAML_AVAILABLE:
        return {
            "valid": False,
            "error": "PyYAML not available - cannot validate YAML"
        }

    try:
        data = yaml.safe_load(yaml_str)
        return {"valid": True, "data": data}
    except yaml.YAMLError as e:
        return {
            "valid": False,
            "error": f"YAML syntax error: {str(e)}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}"
        }


def validate_agent_dag(agents_config: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure delegation relationships form a DAG (no cycles).

    Args:
        agents_config: Agents configuration dict with "agents" key

    Returns:
        dict with keys:
        - valid (bool): True if DAG (no cycles)
        - error (str): Error message if cycle detected (optional)
        - cycle (list): Agent names in cycle path (optional)
    """
    if "agents" not in agents_config:
        return {
            "valid": False,
            "error": "Config missing 'agents' key"
        }

    # Build adjacency list
    graph = {}
    for agent_name, config in agents_config["agents"].items():
        graph[agent_name] = config.get("can_delegate_to", [])

    # DFS cycle detection
    def has_cycle(node: str, visited: set[str], rec_stack: set[str], path: list[str]) -> tuple[bool, list[str]]:
        """
        Detect cycles using DFS.

        Args:
            node: Current node
            visited: Set of all visited nodes
            rec_stack: Set of nodes in current recursion stack
            path: Current path being explored

        Returns:
            (has_cycle, cycle_path)
        """
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle_detected, cycle_path = has_cycle(neighbor, visited, rec_stack, path.copy())
                if cycle_detected:
                    return True, cycle_path
            elif neighbor in rec_stack:
                # Cycle detected - build cycle path
                cycle_start_idx = path.index(neighbor)
                cycle_path = path[cycle_start_idx:] + [neighbor]
                return True, cycle_path

        rec_stack.remove(node)
        return False, []

    # Check each connected component
    visited = set()
    for node in graph:
        if node not in visited:
            has_cycle_detected, cycle_path = has_cycle(node, visited, set(), [])
            if has_cycle_detected:
                return {
                    "valid": False,
                    "error": f"Cycle detected in delegation relationships",
                    "cycle": cycle_path
                }

    return {"valid": True}


def validate_agent_config(config: dict[str, Any], behaviors_dir: Path | None = None) -> dict[str, Any]:
    """
    Validate agent configuration structure and references.

    Checks:
    - Required keys present (role, system_prompt, behaviors)
    - Referenced behaviors exist (if behaviors_dir provided)
    - System prompt is tool-focused (not conversational)
    - Blurb is concise (if present)

    Args:
        config: Agent config dict
        behaviors_dir: Path to behaviors directory (optional, for checking behavior existence)

    Returns:
        dict with keys:
        - valid (bool): True if valid
        - error (str): Error message if invalid (optional)
        - warnings (list): Non-critical warnings (optional)
    """
    warnings = []

    # Check required keys
    required_keys = ["role", "system_prompt", "behaviors"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        return {
            "valid": False,
            "error": f"Missing required keys: {', '.join(missing)}"
        }

    # Check behaviors is a list
    if not isinstance(config["behaviors"], list):
        return {
            "valid": False,
            "error": "behaviors must be a list"
        }

    # Check behavior structure
    for i, behavior in enumerate(config["behaviors"]):
        if not isinstance(behavior, dict):
            return {
                "valid": False,
                "error": f"Behavior {i} must be a dict"
            }
        if "type" not in behavior:
            return {
                "valid": False,
                "error": f"Behavior {i} missing 'type' key"
            }

    # Check if behaviors exist (if behaviors_dir provided)
    if behaviors_dir:
        behaviors_dir = Path(behaviors_dir)
        for behavior in config["behaviors"]:
            behavior_type = behavior["type"]
            # Convert CamelCase to snake_case for file name
            import re
            behavior_file = re.sub(r'(?<!^)(?=[A-Z])', '_', behavior_type).lower()
            behavior_file = behavior_file.replace("_behavior", "") + ".py"
            behavior_path = behaviors_dir / behavior_file

            if not behavior_path.exists():
                warnings.append(f"Behavior file not found: {behavior_path}")

    # Check system prompt is tool-focused (not conversational)
    system_prompt = config.get("system_prompt", "")
    conversational_phrases = [
        "hi!",
        "hello!",
        "i'm here to help",
        "just tell me",
        "i'll try my best"
    ]

    if any(phrase in system_prompt.lower() for phrase in conversational_phrases):
        warnings.append("System prompt appears conversational - should be tool-focused")

    # Check blurb length (if present)
    if "blurb" in config:
        blurb = config["blurb"]
        if len(blurb) > 500:
            warnings.append(f"Blurb is long ({len(blurb)} chars) - should be concise (3-5 sentences)")

    result = {"valid": True}
    if warnings:
        result["warnings"] = warnings

    return result


def validate_delegation_tool(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate delegation_tool structure.

    Args:
        config: Agent config dict

    Returns:
        dict with keys:
        - valid (bool): True if valid
        - error (str): Error message if invalid (optional)
    """
    if "delegation_tool" not in config:
        # Optional - not an error
        return {"valid": True}

    tool = config["delegation_tool"]

    # Check required keys
    required_keys = ["name", "description", "parameters"]
    missing = [k for k in required_keys if k not in tool]
    if missing:
        return {
            "valid": False,
            "error": f"delegation_tool missing keys: {', '.join(missing)}"
        }

    # Check parameters structure
    params = tool.get("parameters", {})
    if not isinstance(params, dict):
        return {
            "valid": False,
            "error": "delegation_tool.parameters must be a dict"
        }

    # Each parameter should have type, description, required
    for param_name, param_spec in params.items():
        if not isinstance(param_spec, dict):
            return {
                "valid": False,
                "error": f"Parameter '{param_name}' must be a dict"
            }

        required_param_keys = ["type", "description", "required"]
        missing = [k for k in required_param_keys if k not in param_spec]
        if missing:
            return {
                "valid": False,
                "error": f"Parameter '{param_name}' missing keys: {', '.join(missing)}"
            }

    return {"valid": True}
