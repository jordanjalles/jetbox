"""
Configuration validation for Jetbox agent configs.

Uses Pydantic for YAML config validation with helpful error messages.
"""

from typing import Any, Optional
from pathlib import Path
import yaml


# Try to import pydantic, provide fallback if not available
try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None
    field_validator = lambda *args, **kwargs: lambda f: f
    ValidationError = Exception


class BehaviorConfig(BaseModel):
    """Configuration for a single behavior."""
    type: str = Field(..., description="Behavior class name (e.g., 'FileToolsBehavior')")
    params: dict[str, Any] = Field(default_factory=dict, description="Behavior parameters")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate behavior type name."""
        if not v:
            raise ValueError("Behavior type cannot be empty")
        if not v[0].isupper():
            raise ValueError(f"Behavior type must start with uppercase: {v}")
        if not v.endswith('Behavior'):
            raise ValueError(f"Behavior type must end with 'Behavior': {v}")
        return v


class DelegationToolParameterConfig(BaseModel):
    """Configuration for a delegation tool parameter."""
    type: str = Field(..., description="Parameter type (e.g., 'string', 'number')")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    enum: Optional[list[str]] = Field(default=None, description="Allowed values (if enum)")


class DelegationToolConfig(BaseModel):
    """Configuration for delegation tool definition."""
    name: str = Field(..., description="Tool name (e.g., 'delegate_to_executor')")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, DelegationToolParameterConfig] = Field(
        default_factory=dict,
        description="Tool parameters"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v:
            raise ValueError("Tool name cannot be empty")
        if ' ' in v:
            raise ValueError(f"Tool name cannot contain spaces: {v}")
        if not v.replace('_', '').isalnum():
            raise ValueError(f"Tool name must be alphanumeric with underscores: {v}")
        return v


class AgentConfig(BaseModel):
    """Configuration schema for agent YAML files."""
    role: str = Field(..., description="Agent role description")
    blurb: Optional[str] = Field(default=None, description="Agent description for parent agents")
    system_prompt: str = Field(..., description="System prompt for the agent")
    behaviors: list[BehaviorConfig] = Field(..., description="List of behaviors to enable")
    delegation_tool: Optional[DelegationToolConfig] = Field(
        default=None,
        description="Delegation tool definition (for delegatable agents)"
    )

    @field_validator('behaviors')
    @classmethod
    def validate_behaviors(cls, v: list[BehaviorConfig]) -> list[BehaviorConfig]:
        """Validate behavior list is not empty."""
        if not v:
            raise ValueError("Agent must have at least one behavior")
        return v

    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        """Validate system prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("System prompt cannot be empty")
        return v


def validate_agent_config(config_path: Path | str) -> tuple[bool, Optional[str], Optional[dict]]:
    """
    Validate an agent configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (is_valid, error_message, config_dict)
        - is_valid: True if config is valid
        - error_message: Error message if invalid, None if valid
        - config_dict: Parsed config dict if valid, None if invalid
    """
    if not PYDANTIC_AVAILABLE:
        # Fallback: basic YAML parsing without validation
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return True, None, config
        except Exception as e:
            return False, f"Failed to parse YAML: {e}", None

    # Pydantic validation
    try:
        # Load YAML
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            return False, "Config file is empty", None

        # Validate with Pydantic
        validated_config = AgentConfig(**config_dict)

        # Return validated data as dict
        return True, None, validated_config.model_dump()

    except FileNotFoundError:
        return False, f"Config file not found: {config_path}", None
    except yaml.YAMLError as e:
        return False, f"Invalid YAML syntax: {e}", None
    except ValidationError as e:
        # Format pydantic validation errors nicely
        error_lines = ["Configuration validation failed:"]
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error['loc'])
            msg = error['msg']
            error_lines.append(f"  • {loc}: {msg}")
        return False, "\n".join(error_lines), None
    except Exception as e:
        return False, f"Unexpected error validating config: {e}", None


def validate_all_agent_configs(base_dir: Path = Path(".")) -> dict[str, tuple[bool, Optional[str]]]:
    """
    Validate all agent config files in the repository.

    Args:
        base_dir: Base directory to search for configs (default: current directory)

    Returns:
        Dict mapping config filename -> (is_valid, error_message)
    """
    results = {}

    # Find all *_config.yaml files
    for config_file in base_dir.glob("*_config.yaml"):
        is_valid, error_msg, _ = validate_agent_config(config_file)
        results[config_file.name] = (is_valid, error_msg)

    return results


def print_validation_report(results: dict[str, tuple[bool, Optional[str]]]) -> None:
    """
    Print a formatted validation report.

    Args:
        results: Results from validate_all_agent_configs()
    """
    print(f"\n{'=' * 60}")
    print("AGENT CONFIG VALIDATION REPORT")
    print(f"{'=' * 60}\n")

    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    total_count = len(results)

    if total_count == 0:
        print("⚠️  No agent config files found")
        return

    print(f"Validated {total_count} config file(s): {valid_count} valid, {total_count - valid_count} invalid\n")

    # Print results
    for config_name, (is_valid, error_msg) in sorted(results.items()):
        status_icon = "✓" if is_valid else "✗"
        status_text = "VALID" if is_valid else "INVALID"

        print(f"{status_icon} {config_name}: {status_text}")
        if error_msg:
            # Indent error message
            for line in error_msg.split('\n'):
                print(f"  {line}")
        print()

    # Summary
    if valid_count == total_count:
        print(f"{'=' * 60}")
        print("✅ All config files are valid!")
        print(f"{'=' * 60}")
    else:
        print(f"{'=' * 60}")
        print(f"❌ {total_count - valid_count} config file(s) have errors")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Validate specific config file
        config_path = Path(sys.argv[1])
        is_valid, error_msg, config = validate_agent_config(config_path)

        if is_valid:
            print(f"✅ {config_path} is valid")
        else:
            print(f"❌ {config_path} is invalid:")
            print(error_msg)
            sys.exit(1)
    else:
        # Validate all configs
        results = validate_all_agent_configs()
        print_validation_report(results)

        # Exit with error code if any invalid
        if any(not is_valid for is_valid, _ in results.values()):
            sys.exit(1)
