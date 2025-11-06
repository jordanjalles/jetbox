from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG = {"rounds": {"max_per_subtask": 12, "max_global": 24}, "hierarchy": {"max_depth": 5, "max_siblings": 8}, "escalation": {"strategy": "force_decompose", "zoom_out_target": "root", "max_approach_retries": 3, "block_failed_paths": True}, "loop_detection": {"max_action_repeats": 3, "max_subtask_repeats": 2, "max_context_age": 300}, "decomposition": {"min_children": 2, "max_children": 6, "temperature": 0.2, "prefer_granular": True}, "approach_retry": {"enabled": True, "reset_subtasks_on_retry": True, "preserve_completed": True, "retry_style": "learn_from_failures"}}

@dataclass
class RoundsConfig:
    max_per_subtask: int
    max_global: int

@dataclass
class HierarchyConfig:
    max_depth: int
    max_siblings: int

@dataclass
class EscalationConfig:
    strategy: str
    zoom_out_target: str
    max_approach_retries: int
    block_failed_paths: bool

@dataclass
class LoopDetectionConfig:
    max_action_repeats: int
    max_subtask_repeats: int
    max_context_age: int

@dataclass
class DecompositionConfig:
    min_children: int
    max_children: int
    temperature: float
    prefer_granular: bool

@dataclass
class ApproachRetryConfig:
    enabled: bool
    reset_subtasks_on_retry: bool
    preserve_completed: bool
    retry_style: str

@dataclass
class LLMTimeoutConfig:
    inactivity_timeout: int
    max_call_time: int
    max_consecutive_timeouts: int
    auto_restart_ollama: bool = False

@dataclass
class LLMConfig:
    model: str
    temperature: float
    system_prompt: str
    max_tokens: int | None = None  # Context window size (num_ctx for Ollama)
    timeout: LLMTimeoutConfig | None = None

@dataclass
class ContextConfig:
    history_keep: int
    max_tokens: int
    recent_actions_limit: int
    enable_compression: bool
    compression_threshold: int

@dataclass
class TimeoutsConfig:
    max_goal_time: int
    create_summary_on_timeout: bool
    save_context_dump: bool

@dataclass
class AgentConfig:
    llm: LLMConfig
    rounds: RoundsConfig
    hierarchy: HierarchyConfig
    escalation: EscalationConfig
    loop_detection: LoopDetectionConfig
    decomposition: DecompositionConfig
    approach_retry: ApproachRetryConfig
    context: ContextConfig
    timeouts: TimeoutsConfig

    @classmethod
    def load(cls, config_path = "agent_config.yaml"):
        import os

        def deep_merge(base: dict, override: dict) -> dict:
            """Deep merge two dicts, recursively merging nested dicts."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        config_dict = DEFAULT_CONFIG.copy()
        config_file = Path(config_path)
        if config_file.exists() and YAML_AVAILABLE:
            try:
                with open(config_file) as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict = deep_merge(config_dict, yaml_config)
            except (yaml.YAMLError, IOError, OSError) as e:
                print(f"Warning: Failed to load {config_file}: {e}")
            except Exception as e:
                print(f"Unexpected error loading {config_file}: {e}")

        # Provide defaults for new sections if not in config
        if "llm" not in config_dict:
            config_dict["llm"] = {
                "model": "qwen3:8b",
                "temperature": 0.2,
                "system_prompt": "You are a coding agent."
            }

        # Allow environment variable override for model
        if "OLLAMA_MODEL" in os.environ:
            config_dict["llm"]["model"] = os.environ["OLLAMA_MODEL"]

        # Add default timeout config if not present
        if "timeout" not in config_dict["llm"]:
            config_dict["llm"]["timeout"] = {
                "inactivity_timeout": 30,
                "max_call_time": 180,
                "max_consecutive_timeouts": 3
            }

        if "context" not in config_dict:
            config_dict["context"] = {
                "history_keep": 12,
                "max_tokens": 8000,
                "recent_actions_limit": 10,
                "enable_compression": False,
                "compression_threshold": 20
            }

        if "timeouts" not in config_dict:
            config_dict["timeouts"] = {
                "max_goal_time": 600,  # 10 minutes
                "create_summary_on_timeout": True,
                "save_context_dump": True
            }

        # Build LLMConfig with timeout sub-config
        llm_dict = config_dict["llm"].copy()
        timeout_dict = llm_dict.pop("timeout", None)
        llm_config = LLMConfig(
            **llm_dict,
            timeout=LLMTimeoutConfig(**timeout_dict) if timeout_dict else None
        )

        return cls(
            llm=llm_config,
            rounds=RoundsConfig(**config_dict["rounds"]),
            hierarchy=HierarchyConfig(**config_dict["hierarchy"]),
            escalation=EscalationConfig(**config_dict["escalation"]),
            loop_detection=LoopDetectionConfig(**config_dict["loop_detection"]),
            decomposition=DecompositionConfig(**config_dict["decomposition"]),
            approach_retry=ApproachRetryConfig(**config_dict["approach_retry"]),
            context=ContextConfig(**config_dict["context"]),
            timeouts=TimeoutsConfig(**config_dict["timeouts"]),
        )


# ============================================================================
# New Config System - Multi-file Structure
# ============================================================================

def load_behavior_defaults() -> dict:
    """
    Load behavior parameter defaults from config/behavior_defaults.yaml.

    Returns:
        Dictionary of behavior parameter defaults
    """
    config_path = Path("config/behavior_defaults.yaml")

    if not config_path.exists():
        return {}

    if not YAML_AVAILABLE:
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load {config_path}: {e}")
        return {}


def load_llm_config() -> dict:
    """
    Load LLM configuration from config/llm_config.yaml.

    Returns:
        Dictionary of LLM configuration (model, temperature, timeout, system_prompt)
    """
    import os

    config_path = Path("config/llm_config.yaml")

    # Defaults if file doesn't exist or YAML not available
    defaults = {
        "model": os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
        "temperature": 0.2,
        "system_prompt": "You are a coding agent.",
        "timeout": {
            "inactivity_timeout": 30,
            "max_call_time": 180,
            "max_consecutive_timeouts": 3,
            "auto_restart_ollama": True
        }
    }

    if not config_path.exists() or not YAML_AVAILABLE:
        return defaults

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
            # Allow environment variable override
            if "OLLAMA_MODEL" in os.environ:
                config["model"] = os.environ["OLLAMA_MODEL"]
            return config
    except Exception as e:
        print(f"Warning: Failed to load {config_path}: {e}")
        return defaults


def load_runtime_config() -> dict:
    """
    Load runtime configuration from config/agent_runtime.yaml.

    Returns:
        Dictionary with rounds, timeouts, hierarchy, escalation, loop_detection,
        decomposition, approach_retry, context sections
    """
    config_path = Path("config/agent_runtime.yaml")

    if not config_path.exists() or not YAML_AVAILABLE:
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Warning: Failed to load {config_path}: {e}")
        return DEFAULT_CONFIG.copy()


def list_available_teams() -> list[dict]:
    """
    List all available team configurations.

    Returns:
        List of dicts with keys: name, file, description, agents
    """
    teams_dir = Path("config/teams")

    if not teams_dir.exists() or not YAML_AVAILABLE:
        return []

    teams = []
    for team_file in sorted(teams_dir.glob("*.yaml")):
        try:
            with open(team_file) as f:
                team_config = yaml.safe_load(f)
                if not team_config:
                    continue

                teams.append({
                    "name": team_config.get("name", team_file.stem),
                    "file": team_file.stem,
                    "description": team_config.get("description", ""),
                    "agents": list(team_config.get("agents", {}).keys())
                })
        except Exception as e:
            print(f"Warning: Failed to load team config {team_file}: {e}")

    return teams


def load_team_config(team_name: str = "default") -> dict:
    """
    Load a team configuration by name.

    Args:
        team_name: Name of team file (without .yaml extension)

    Returns:
        Dictionary with team configuration (name, description, agents)
    """
    config_path = Path(f"config/teams/{team_name}.yaml")

    if not config_path.exists() or not YAML_AVAILABLE:
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load team config {config_path}: {e}")
        return {}


def load_agent_config(agent_name: str) -> dict:
    """
    Load an individual agent configuration by name.

    Args:
        agent_name: Name of agent config file (without .yaml extension)

    Returns:
        Dictionary with agent configuration (role, blurb, delegation_tool,
        system_prompt, behaviors)
    """
    config_path = Path(f"config/agents/{agent_name}.yaml")

    if not config_path.exists() or not YAML_AVAILABLE:
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load agent config {config_path}: {e}")
        return {}


# Keep backward compatibility - load from old location by default
config = AgentConfig.load()
