"""
Validation utilities for agent configuration generation.

This module provides functions to validate generated agent configs:
- validate_yaml_syntax: Parse YAML
- validate_agent_config_structure: Check required sections
- validate_behavior_references: Check behaviors exist
- validate_agent_dag: Detect delegation cycles
- validate_delegation_tool_schema: Validate delegation tool

These validators are used by CreateAgentBehavior to ensure
generated agent configs are valid and safe.
"""

import yaml
from pathlib import Path
from typing import Any


# Known behavior types (from behaviors directory)
KNOWN_BEHAVIOR_TYPES = [
    "ArchitectToolsBehavior",
    "ChatbotBehavior",
    "CommandToolsBehavior",
    "CompactWhenNearFullBehavior",
    "DelegationBehavior",
    "DirectoryToolsBehavior",
    "LoopDetectionBehavior",
    "ReadFileToolsBehavior",
    "ServerManagementBehavior",
    "ServerToolsBehavior",
    "StatusDisplayBehavior",
    "TaskManagementBehavior",
    "WorkspaceManagementBehavior",
    "WorkspaceTaskNotesBehavior",
    "WriteFileToolsBehavior"
]


def validate_yaml_syntax(yaml_content: str = None, file_path: str = None) -> dict[str, Any]:
    """
    Parse YAML and validate syntax.

    Args:
        yaml_content: YAML string to parse (optional if file_path provided)
        file_path: Path to YAML file (optional if yaml_content provided)

    Returns:
        Dictionary with:
        - valid: True if YAML is valid, False otherwise
        - error: Error message if invalid (optional)
        - data: Parsed YAML data if valid (optional)

    Example:
        >>> result = validate_yaml_syntax("key: value")
        >>> assert result["valid"] is True
        >>> assert result["data"] == {"key": "value"}
    """
    # Read from file if provided
    if file_path:
        try:
            yaml_content = Path(file_path).read_text()
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to read file: {str(e)}"
            }

    if not yaml_content:
        return {
            "valid": False,
            "error": "No YAML content or file path provided"
        }

    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
        return {"valid": True, "data": data}
    except yaml.YAMLError as e:
        return {
            "valid": False,
            "error": f"YAML syntax error: {str(e)}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Parse error: {str(e)}"
        }


def validate_agent_config_structure(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate agent configuration has required sections.

    Required sections:
    - role: Human-readable role description
    - blurb: Description for parent agents
    - system_prompt: Instructions for agent
    - behaviors: List of behavior configurations

    Args:
        config: Agent configuration dict

    Returns:
        Dictionary with:
        - valid: True if structure is valid, False otherwise
        - error: Error message if invalid (optional)
        - warnings: List of warnings (optional)

    Example:
        >>> config = {
        ...     "role": "Coder",
        ...     "blurb": "Does coding",
        ...     "system_prompt": "You code",
        ...     "behaviors": []
        ... }
        >>> result = validate_agent_config_structure(config)
        >>> assert result["valid"] is True
    """
    if not isinstance(config, dict):
        return {
            "valid": False,
            "error": "Config must be a dictionary"
        }

    # Required keys
    required_keys = ["role", "blurb", "system_prompt", "behaviors"]
    missing_keys = [k for k in required_keys if k not in config]

    if missing_keys:
        return {
            "valid": False,
            "error": f"Missing required keys: {', '.join(missing_keys)}"
        }

    warnings = []

    # Validate role
    role = config.get("role")
    if not isinstance(role, str) or not role.strip():
        return {
            "valid": False,
            "error": "Role must be a non-empty string"
        }

    # Validate blurb
    blurb = config.get("blurb")
    if not isinstance(blurb, str) or not blurb.strip():
        return {
            "valid": False,
            "error": "Blurb must be a non-empty string"
        }

    # Check blurb length (should be 3-5 sentences)
    sentences = blurb.count('.') + blurb.count('!') + blurb.count('?')
    if sentences < 2:
        warnings.append("Blurb is very short (< 2 sentences)")
    elif sentences > 10:
        warnings.append("Blurb is very long (> 10 sentences)")

    # Validate system_prompt
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return {
            "valid": False,
            "error": "System prompt must be a non-empty string"
        }

    # Check if system prompt is too conversational
    conversational_phrases = [
        "hi!", "hello!", "i'm here to help", "just tell me",
        "i'll try my best", "happy to help"
    ]
    lower_prompt = system_prompt.lower()
    if any(phrase in lower_prompt for phrase in conversational_phrases):
        warnings.append("System prompt may be too conversational (should be tool-focused)")

    # Validate behaviors list
    behaviors = config.get("behaviors")
    if not isinstance(behaviors, list):
        return {
            "valid": False,
            "error": "Behaviors must be a list"
        }

    if len(behaviors) == 0:
        warnings.append("No behaviors configured (agent will have no capabilities)")

    # Check each behavior entry
    for i, behavior in enumerate(behaviors):
        if not isinstance(behavior, dict):
            return {
                "valid": False,
                "error": f"Behavior at index {i} must be a dictionary"
            }

        if "type" not in behavior:
            return {
                "valid": False,
                "error": f"Behavior at index {i} missing 'type' key"
            }

        if not isinstance(behavior["type"], str):
            return {
                "valid": False,
                "error": f"Behavior at index {i} type must be a string"
            }

    # Validate delegation_tool if present
    if "delegation_tool" in config:
        delegation_tool = config["delegation_tool"]
        if not isinstance(delegation_tool, dict):
            return {
                "valid": False,
                "error": "delegation_tool must be a dictionary"
            }

        required_tool_keys = ["name", "description", "parameters"]
        missing_tool_keys = [k for k in required_tool_keys if k not in delegation_tool]
        if missing_tool_keys:
            return {
                "valid": False,
                "error": f"delegation_tool missing keys: {', '.join(missing_tool_keys)}"
            }

    result = {"valid": True}
    if warnings:
        result["warnings"] = warnings

    return result


def validate_behavior_references(config: dict[str, Any]) -> dict[str, Any]:
    """
    Check that all referenced behaviors exist.

    Args:
        config: Agent configuration dict

    Returns:
        Dictionary with:
        - valid: True if all behaviors exist, False otherwise
        - error: Error message if invalid (optional)
        - unknown_behaviors: List of unknown behavior types (optional)

    Example:
        >>> config = {
        ...     "behaviors": [
        ...         {"type": "ChatbotBehavior"},
        ...         {"type": "UnknownBehavior"}
        ...     ]
        ... }
        >>> result = validate_behavior_references(config)
        >>> assert result["valid"] is False
        >>> assert "UnknownBehavior" in result["unknown_behaviors"]
    """
    if "behaviors" not in config:
        return {
            "valid": False,
            "error": "Config missing 'behaviors' key"
        }

    behaviors = config["behaviors"]
    if not isinstance(behaviors, list):
        return {
            "valid": False,
            "error": "Behaviors must be a list"
        }

    unknown_behaviors = []

    for behavior in behaviors:
        if not isinstance(behavior, dict):
            continue

        behavior_type = behavior.get("type")
        if not behavior_type:
            continue

        # Check if behavior type is known
        if behavior_type not in KNOWN_BEHAVIOR_TYPES:
            unknown_behaviors.append(behavior_type)

    if unknown_behaviors:
        return {
            "valid": False,
            "error": f"Unknown behavior types referenced: {', '.join(unknown_behaviors)}",
            "unknown_behaviors": unknown_behaviors
        }

    return {"valid": True}


def validate_agent_dag(agents_config: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure delegation relationships form a DAG (no cycles).

    Cycles in delegation create infinite loops, so we must detect them.

    Args:
        agents_config: Agents configuration dict with structure:
            {
                "agents": {
                    "agent_name": {
                        "can_delegate_to": ["other_agent", ...]
                    }
                }
            }

    Returns:
        Dictionary with:
        - valid: True if no cycles, False if cycle detected
        - error: Error message if cycle detected (optional)
        - cycle: List of agents involved in cycle (optional)

    Example:
        >>> config = {
        ...     "agents": {
        ...         "a": {"can_delegate_to": ["b"]},
        ...         "b": {"can_delegate_to": ["c"]},
        ...         "c": {"can_delegate_to": []}
        ...     }
        ... }
        >>> result = validate_agent_dag(config)
        >>> assert result["valid"] is True
        >>>
        >>> config = {
        ...     "agents": {
        ...         "a": {"can_delegate_to": ["b"]},
        ...         "b": {"can_delegate_to": ["a"]}
        ...     }
        ... }
        >>> result = validate_agent_dag(config)
        >>> assert result["valid"] is False
    """
    if "agents" not in agents_config:
        return {
            "valid": False,
            "error": "Config missing 'agents' key"
        }

    agents = agents_config["agents"]
    if not isinstance(agents, dict):
        return {
            "valid": False,
            "error": "Agents must be a dictionary"
        }

    # Build adjacency list
    graph = {}
    for agent_name, config in agents.items():
        if not isinstance(config, dict):
            continue

        can_delegate = config.get("can_delegate_to", [])
        if not isinstance(can_delegate, list):
            can_delegate = []

        graph[agent_name] = can_delegate

    # DFS cycle detection
    def has_cycle(node, visited, rec_stack, path):
        """
        Detect cycle using DFS.

        Args:
            node: Current node
            visited: Set of visited nodes
            rec_stack: Set of nodes in current recursion stack
            path: Current path (for cycle reporting)

        Returns:
            Tuple of (has_cycle: bool, cycle_path: list)
        """
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Check neighbors
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle_detected, cycle_path = has_cycle(neighbor, visited, rec_stack, path[:])
                if cycle_detected:
                    return True, cycle_path
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start_idx = path.index(neighbor)
                cycle = path[cycle_start_idx:] + [neighbor]
                return True, cycle

        rec_stack.remove(node)
        return False, []

    # Check all nodes
    visited = set()
    for node in graph:
        if node not in visited:
            cycle_detected, cycle_path = has_cycle(node, visited, set(), [])
            if cycle_detected:
                return {
                    "valid": False,
                    "error": f"Delegation cycle detected: {' -> '.join(cycle_path)}",
                    "cycle": cycle_path
                }

    return {"valid": True}


def validate_delegation_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Validate delegation tool schema.

    Similar to validate_tool_schema but with delegation-specific checks.

    Args:
        tool: Delegation tool definition dict

    Returns:
        Dictionary with:
        - valid: True if schema is valid, False otherwise
        - error: Error message if invalid (optional)

    Example:
        >>> tool = {
        ...     "name": "delegate_to_executor",
        ...     "description": "Delegate task",
        ...     "parameters": {
        ...         "task_description": {
        ...             "type": "string",
        ...             "description": "Task",
        ...             "required": True
        ...         }
        ...     }
        ... }
        >>> result = validate_delegation_tool_schema(tool)
        >>> assert result["valid"] is True
    """
    if not isinstance(tool, dict):
        return {"valid": False, "error": "Tool must be a dictionary"}

    required_keys = ["name", "description", "parameters"]
    missing_keys = [k for k in required_keys if k not in tool]

    if missing_keys:
        return {
            "valid": False,
            "error": f"Missing required keys: {', '.join(missing_keys)}"
        }

    # Validate name
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "valid": False,
            "error": "Tool name must be a non-empty string"
        }

    # Check name format (should start with "delegate_")
    if not name.startswith("delegate_"):
        return {
            "valid": False,
            "error": f"Delegation tool name should start with 'delegate_': {name}"
        }

    # Validate description
    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        return {
            "valid": False,
            "error": "Tool description must be a non-empty string"
        }

    # Validate parameters
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return {
            "valid": False,
            "error": "Tool parameters must be a dictionary"
        }

    if len(parameters) == 0:
        return {
            "valid": False,
            "error": "Tool must have at least one parameter"
        }

    # Check each parameter
    for param_name, param_spec in parameters.items():
        if not isinstance(param_spec, dict):
            return {
                "valid": False,
                "error": f"Parameter '{param_name}' spec must be a dictionary"
            }

        required_param_keys = ["type", "description"]
        missing_param_keys = [k for k in required_param_keys if k not in param_spec]

        if missing_param_keys:
            return {
                "valid": False,
                "error": f"Parameter '{param_name}' missing keys: {', '.join(missing_param_keys)}"
            }

    return {"valid": True}
