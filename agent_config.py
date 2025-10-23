from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG = {"rounds": {"max_per_subtask": 12, "max_per_task": 128, "max_global": 24}, "hierarchy": {"max_depth": 5, "max_siblings": 8}, "escalation": {"strategy": "force_decompose", "zoom_out_target": "root", "max_approach_retries": 3, "block_failed_paths": True}, "loop_detection": {"max_action_repeats": 3, "max_subtask_repeats": 2, "max_context_age": 300}, "decomposition": {"min_children": 2, "max_children": 6, "temperature": 0.2, "prefer_granular": True}, "approach_retry": {"enabled": True, "reset_subtasks_on_retry": True, "preserve_completed": True, "retry_style": "learn_from_failures"}}

@dataclass
class RoundsConfig:
    max_per_subtask: int
    max_per_task: int
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
class ContextConfig:
    max_messages_in_memory: int
    max_tokens: int
    recent_actions_limit: int
    enable_compression: bool
    compression_threshold: int

@dataclass
class AgentConfig:
    rounds: RoundsConfig
    hierarchy: HierarchyConfig
    escalation: EscalationConfig
    loop_detection: LoopDetectionConfig
    decomposition: DecompositionConfig
    approach_retry: ApproachRetryConfig
    context: ContextConfig

    @classmethod
    def load(cls, config_path = "agent_config.yaml"):
        config_dict = DEFAULT_CONFIG.copy()
        config_file = Path(config_path)
        if config_file.exists() and YAML_AVAILABLE:
            try:
                with open(config_file) as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict = {**config_dict, **yaml_config}
            except: pass
        # Provide default for context if not in config
        if "context" not in config_dict:
            config_dict["context"] = {
                "max_messages_in_memory": 12,
                "max_tokens": 8000,
                "recent_actions_limit": 10,
                "enable_compression": True,
                "compression_threshold": 20
            }
        return cls(
            rounds=RoundsConfig(**config_dict["rounds"]),
            hierarchy=HierarchyConfig(**config_dict["hierarchy"]),
            escalation=EscalationConfig(**config_dict["escalation"]),
            loop_detection=LoopDetectionConfig(**config_dict["loop_detection"]),
            decomposition=DecompositionConfig(**config_dict["decomposition"]),
            approach_retry=ApproachRetryConfig(**config_dict["approach_retry"]),
            context=ContextConfig(**config_dict["context"]),
        )

config = AgentConfig.load()
