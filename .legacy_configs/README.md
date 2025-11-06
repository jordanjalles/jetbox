# Legacy Config Files

This directory contains deprecated configuration files from the old config system.

## Status: DEPRECATED

These files are no longer used by the agent system. They are kept here for reference only.

## Old Config Structure (Deprecated)

The old system used flat config files in the root directory:
- `agent_config.yaml` - Behavior defaults, LLM config, and runtime config (all in one file)
- `agents.yaml` - Team configuration
- `{agent_name}_config.yaml` - Individual agent configs (e.g., `orchestrator_config.yaml`)

## New Config Structure (Current)

The new system uses a structured config directory:

```
config/
├── behavior_defaults.yaml      # Global behavior parameter defaults
├── llm_config.yaml             # LLM settings (model, temperature, timeouts)
├── agent_runtime.yaml          # Runtime config (rounds, hierarchy, escalation)
├── agents/
│   ├── orchestrator.yaml       # Orchestrator agent config
│   ├── task_executor.yaml      # Task executor agent config
│   └── architect.yaml          # Architect agent config
└── teams/
    ├── default.yaml            # Default team configuration
    └── solo.yaml               # Solo agent team configuration
```

## Migration

If you need to migrate custom configs to the new structure:

1. **Behavior defaults**: Extract `behavior_defaults` section from old `agent_config.yaml` → `config/behavior_defaults.yaml`
2. **LLM config**: Extract `llm` section from old `agent_config.yaml` → `config/llm_config.yaml`
3. **Runtime config**: Extract runtime sections (rounds, hierarchy, escalation, etc.) → `config/agent_runtime.yaml`
4. **Team config**: Move `agents.yaml` → `config/teams/default.yaml`
5. **Agent configs**: Move `{agent}_config.yaml` → `config/agents/{agent}.yaml`

## Documentation

For complete documentation on the new config system, see:
- [CONFIG_QUICK_START.md](../CONFIG_QUICK_START.md)
- [CONFIG_REFACTOR_PLAN.md](../CONFIG_REFACTOR_PLAN.md)
- [CONFIG_REFACTOR_IMPLEMENTATION_SUMMARY.md](../CONFIG_REFACTOR_IMPLEMENTATION_SUMMARY.md)

## When to Delete

These legacy files can be safely deleted once you've verified:
1. All your custom configs have been migrated to `config/`
2. All tests pass with the new config structure
3. All agents instantiate correctly

## Do NOT Use

**WARNING**: The agent system NO LONGER reads these files. Any changes made here will have NO EFFECT.

Use the files in `config/` instead.
