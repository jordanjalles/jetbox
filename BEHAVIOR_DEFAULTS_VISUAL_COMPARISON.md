# Behavior Defaults Refactor - Visual Comparison

## Before: Duplicated Parameters Across All Agent Configs

### agent_config.yaml (BEFORE)
```yaml
# Agent Configuration
# No behavior_defaults section

# ==============================
# LLM Settings
# ==============================
llm:
  model: "qwen3:14b"
  temperature: 0.2
```

### task_executor_config.yaml (BEFORE)
```yaml
behaviors:
  - type: SubAgentContextBehavior
    params: {}

  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 8000          # ← DUPLICATED
      compact_threshold: 0.75   # ← DUPLICATED
      keep_recent: 20           # ← DUPLICATED

  - type: CommandToolsBehavior
    params:
      whitelist:                # ← DUPLICATED
        - python
        - pytest
        - ruff
        - pip

  - type: LoopDetectionBehavior
    params:
      max_repeats: 5            # ← DUPLICATED
```

### orchestrator_config.yaml (BEFORE)
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 8000          # ← DUPLICATED
      compact_threshold: 0.75   # ← DUPLICATED

  - type: LoopDetectionBehavior
    params:
      max_repeats: 5            # ← DUPLICATED
```

### architect_config.yaml (BEFORE)
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 8000          # ← DUPLICATED
      compact_threshold: 0.75   # ← DUPLICATED

  - type: LoopDetectionBehavior
    params:
      max_repeats: 5            # ← DUPLICATED
```

**Problem**: To change `max_tokens` for a new model, need to update 3 files.

---

## After: Global Defaults with Clean Agent Configs

### agent_config.yaml (AFTER)
```yaml
# Agent Configuration

# ==============================
# Global Behavior Parameter Defaults
# ==============================
behavior_defaults:
  CompactWhenNearFullBehavior:
    max_tokens: 8000          # ← SINGLE SOURCE OF TRUTH
    compact_threshold: 0.75   # ← SINGLE SOURCE OF TRUTH
    keep_recent: 20           # ← SINGLE SOURCE OF TRUTH

  LoopDetectionBehavior:
    max_repeats: 5            # ← SINGLE SOURCE OF TRUTH

  CommandToolsBehavior:
    whitelist:                # ← SINGLE SOURCE OF TRUTH
      - python
      - pytest
      - ruff
      - pip

  # ... other behaviors

# ==============================
# LLM Settings
# ==============================
llm:
  model: "qwen3:14b"
  temperature: 0.2
```

### task_executor_config.yaml (AFTER)
```yaml
behaviors:
  - type: SubAgentContextBehavior
  - type: CompactWhenNearFullBehavior    # Uses global defaults
  - type: CommandToolsBehavior           # Uses global defaults
  - type: LoopDetectionBehavior          # Uses global defaults
```

### orchestrator_config.yaml (AFTER)
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior    # Uses global defaults
  - type: LoopDetectionBehavior          # Uses global defaults
```

### architect_config.yaml (AFTER)
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior    # Uses global defaults
  - type: LoopDetectionBehavior          # Uses global defaults
```

**Solution**: To change `max_tokens` for a new model, update 1 file (agent_config.yaml).

---

## Code Changes: base_agent.py

### Before
```python
def load_behaviors_from_config(self, config_file: str) -> None:
    # ... setup code ...

    for behavior_spec in config.get("behaviors", []):
        behavior_type = behavior_spec["type"]
        behavior_params = behavior_spec.get("params", {})  # ← Just use agent params

        try:
            behavior_class = self._import_behavior_class(behavior_type)
            behavior = behavior_class(**behavior_params)
            self.add_behavior(behavior)
            print(f"[{self.name}] Loaded behavior: {behavior_type}")
        except Exception as e:
            print(f"[{self.name}] Failed to load behavior {behavior_type}: {e}")
```

### After
```python
def load_behaviors_from_config(self, config_file: str) -> None:
    # ... setup code ...

    # Load global behavior defaults
    global_defaults = self._load_global_behavior_defaults()  # ← NEW

    for behavior_spec in config.get("behaviors", []):
        behavior_type = behavior_spec["type"]

        # Get global defaults for this behavior type
        default_params = global_defaults.get(behavior_type, {})  # ← NEW
        if default_params is None:
            default_params = {}

        # Get agent-specific overrides
        agent_params = behavior_spec.get("params", {})
        if agent_params is None:
            agent_params = {}

        # Merge: agent params override global defaults
        behavior_params = {**default_params, **agent_params}  # ← NEW

        try:
            behavior_class = self._import_behavior_class(behavior_type)
            behavior = behavior_class(**behavior_params)
            self.add_behavior(behavior)

            # Log parameter source
            if agent_params:
                print(f"[{self.name}] Loaded behavior: {behavior_type} (agent-specific params: {agent_params})")
            elif default_params:
                print(f"[{self.name}] Loaded behavior: {behavior_type} (using global defaults)")  # ← NEW
            else:
                print(f"[{self.name}] Loaded behavior: {behavior_type} (no parameters)")
        except Exception as e:
            print(f"[{self.name}] Failed to load behavior {behavior_type}: {e}")

def _load_global_behavior_defaults(self) -> dict[str, dict[str, Any]]:  # ← NEW METHOD
    """Load global behavior parameter defaults from agent_config.yaml."""
    import yaml
    config_path = Path(__file__).parent / "agent_config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("behavior_defaults", {})
    except Exception as e:
        print(f"[{self.name}] Warning: Failed to load global behavior defaults: {e}")
        return {}
```

---

## Output Comparison

### Before
```
[task_executor] Loading behaviors from task_executor_config.yaml
[task_executor] Loaded behavior: SubAgentContextBehavior
[task_executor] Loaded behavior: CompactWhenNearFullBehavior
[task_executor] Loaded behavior: CommandToolsBehavior
[task_executor] Loaded behavior: LoopDetectionBehavior
```

### After
```
[task_executor] Loading behaviors from task_executor_config.yaml
[task_executor] Loaded behavior: SubAgentContextBehavior (no parameters)
[task_executor] Loaded behavior: CompactWhenNearFullBehavior (using global defaults)
[task_executor] Loaded behavior: CommandToolsBehavior (using global defaults)
[task_executor] Loaded behavior: LoopDetectionBehavior (using global defaults)
```

**Improvement**: Clear visibility into where parameters come from.

---

## Override Example

If an agent needs different parameters, it can still override:

### orchestrator_config.yaml (with override)
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
    # Uses global defaults

  - type: LoopDetectionBehavior
    params:
      max_repeats: 10  # Override: orchestrator gets more loop tolerance
```

### Output
```
[orchestrator] Loaded behavior: CompactWhenNearFullBehavior (using global defaults)
[orchestrator] Loaded behavior: LoopDetectionBehavior (agent-specific params: {'max_repeats': 10})
```

**Result**: `max_repeats=10` for orchestrator, `max_repeats=5` for other agents.

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Files to edit for model change** | 3 files | 1 file |
| **Parameter duplication** | 3x duplicated | 0x (DRY) |
| **Consistency risk** | High (manual sync) | Low (single source) |
| **Discoverability** | Spread across files | One location |
| **Flexibility** | None (all hardcoded) | Can override per-agent |
| **Visibility** | No indication of source | Clear logging |

---

## Test Results

```bash
$ python test_global_behavior_defaults.py

======================================================================
Testing Global Behavior Defaults System
======================================================================

=== Test 1: Global defaults loaded from agent_config.yaml ===
✓ Found behavior_defaults section with 10 behavior types
✓ CompactWhenNearFullBehavior: {'max_tokens': 8000, ...}
✓ LoopDetectionBehavior: {'max_repeats': 5}
✓ CommandToolsBehavior: {'whitelist': ['python', 'pytest', 'ruff', 'pip']}
✓ All expected behavior defaults found

=== Test 2: TaskExecutor behavior parameters ===
✓ CompactWhenNearFullBehavior.max_tokens = 8000
✓ CompactWhenNearFullBehavior.compact_threshold = 0.75
✓ CompactWhenNearFullBehavior.keep_recent = 20
✓ LoopDetectionBehavior.max_repeats = 5
✓ All TaskExecutor behavior parameters correct

=== Test 3: Orchestrator behavior parameters ===
✓ CompactWhenNearFullBehavior.max_tokens = 8000
✓ CompactWhenNearFullBehavior.compact_threshold = 0.75
✓ LoopDetectionBehavior.max_repeats = 5
✓ All Orchestrator behavior parameters correct

=== Test 4: Architect behavior parameters ===
✓ CompactWhenNearFullBehavior.max_tokens = 8000
✓ CompactWhenNearFullBehavior.compact_threshold = 0.75
✓ LoopDetectionBehavior.max_repeats = 5
✓ All Architect behavior parameters correct

=== Test 5: Override mechanism ===
✓ LoopDetectionBehavior.max_repeats = 10 (overridden)
✓ CompactWhenNearFullBehavior.max_tokens = 16000 (overridden)
✓ Override mechanism works correctly

======================================================================
Test Summary
======================================================================
✓ PASS: Global defaults loaded
✓ PASS: TaskExecutor parameters
✓ PASS: Orchestrator parameters
✓ PASS: Architect parameters
✓ PASS: Override mechanism

Total: 5/5 tests passed

🎉 All tests passed! Global behavior defaults system working correctly.
```

---

## Migration Path

**Current agents**: No changes needed - already using global defaults.

**New behaviors**: Add to `behavior_defaults` in `agent_config.yaml`.

**Agent-specific needs**: Add `params:` override in agent config.

**Model changes**: Update `max_tokens` in one place (agent_config.yaml).

---

## Conclusion

This refactor achieves:
- ✅ DRY compliance (no duplication)
- ✅ Single source of truth for parameters
- ✅ Easy model switching (1 file update)
- ✅ Maintained flexibility (can override)
- ✅ Better visibility (clear logging)
- ✅ Zero breaking changes (100% compatible)
