# Architect → Executor Coordination Proposals

**Problem:** Task executor agents ignore architect artifacts (task-breakdown.json, architecture docs, module specs) and implement from scratch, leading to wasted architecture work and inconsistent results.

**Current State:**
- Architect creates: `architecture/task-breakdown.json`, `architecture/system-overview.md`, `architecture/modules/*.md`
- Orchestrator delegates with: "follow the architecture and module specs provided by the architect"
- Task executor: Starts implementing immediately without reading any artifacts

## Proposal 1: Artifact Loading Behavior (RECOMMENDED)

**Approach:** Create a composable `ArchitectArtifactsBehavior` that automatically loads and injects architect artifacts into task_executor context.

### Implementation

```python
# behaviors/architect_artifacts.py
class ArchitectArtifactsBehavior(AgentBehavior):
    """
    Auto-loads architect artifacts when present in workspace.

    Checks for:
    - architecture/task-breakdown.json
    - architecture/system-overview.md (or first .md in architecture/)
    - Relevant module specs from architecture/modules/

    Injects into context as structured information.
    """

    def enhance_context(self, context, **kwargs):
        """Load and inject architect artifacts."""
        workspace_manager = kwargs.get('workspace_manager')

        if not workspace_manager:
            return context

        artifacts = self._load_artifacts(workspace_manager)

        if not artifacts:
            return context

        # Build artifact context
        context_parts = []
        context_parts.append("ARCHITECT ARTIFACTS:")
        context_parts.append("")

        # 1. High-level overview
        if artifacts.get('overview'):
            context_parts.append("## System Overview")
            context_parts.append(artifacts['overview'][:1000])  # Truncate
            context_parts.append("")

        # 2. Task breakdown (most important!)
        if artifacts.get('tasks'):
            context_parts.append("## Task Breakdown")
            context_parts.append(f"Total tasks: {len(artifacts['tasks'])}")
            for task in artifacts['tasks']:
                status_icon = "✓" if task['status'] == 'completed' else "○"
                deps = f" (depends on: {', '.join(task['dependencies'])})" if task['dependencies'] else ""
                context_parts.append(f"  {status_icon} {task['task_id']}: {task['description']}{deps}")
            context_parts.append("")

        # 3. Module references
        if artifacts.get('modules'):
            context_parts.append(f"## Module Specs")
            context_parts.append(f"{len(artifacts['modules'])} module specs available in architecture/modules/")
            context_parts.append("")

        context_parts.append("IMPORTANT: Follow the task breakdown above. Read relevant module specs before implementing.")

        # Insert after goal, before chat history
        context.insert(2, {
            "role": "user",
            "content": "\n".join(context_parts)
        })

        return context

    def _load_artifacts(self, workspace_manager):
        """Load all architect artifacts from workspace."""
        artifacts = {}
        arch_dir = workspace_manager.workspace_path / "architecture"

        if not arch_dir.exists():
            return artifacts

        # Load task-breakdown.json
        task_file = arch_dir / "task-breakdown.json"
        if task_file.exists():
            with open(task_file) as f:
                task_data = json.load(f)
                artifacts['tasks'] = task_data.get('tasks', [])

        # Load overview (first .md in architecture/)
        md_files = list(arch_dir.glob("*.md"))
        if md_files:
            with open(md_files[0]) as f:
                artifacts['overview'] = f.read()

        # Check for module specs
        modules_dir = arch_dir / "modules"
        if modules_dir.exists():
            artifacts['modules'] = list(modules_dir.glob("*.md"))

        return artifacts
```

**Configuration:**
```yaml
# task_executor_config.yaml
behaviors:
  # ... existing behaviors ...

  # Artifact coordination (NEW)
  - type: ArchitectArtifactsBehavior  # Auto-loads architect artifacts
```

**Prompt Enhancement:**
```yaml
# task_executor_config.yaml
system_prompt: |
  ...existing prompt...

  ARCHITECTURE COORDINATION:
  - If architect artifacts are present (task breakdown, module specs), follow them
  - Read relevant module specs from architecture/modules/ before implementing
  - Update task status in task-breakdown.json as you complete tasks
  - Use list_dir to check for architecture/ directory on startup
```

### Pros
- ✅ Automatic - no LLM reasoning required
- ✅ Composable - just add behavior to config
- ✅ Reliable - artifacts always loaded if present
- ✅ Minimal context overhead (summary only, not full files)
- ✅ Works with existing architect workflow

### Cons
- ❌ Context injection happens even if task_executor won't use it
- ❌ No explicit task-by-task execution (just context awareness)
- ❌ Requires new behavior implementation

### Effort
- Medium: ~200 lines of code, unit tests, integration test
- 2-3 hours to implement and test

---

## Proposal 2: Orchestrator Task Distribution

**Approach:** Orchestrator reads task-breakdown.json and delegates individual tasks to task_executor one at a time.

### Implementation

```python
# behaviors/delegation.py - enhance existing DelegationBehavior

def get_tools(self):
    """Add new orchestrator tool."""
    return [
        # ... existing tools ...
        {
            "type": "function",
            "function": {
                "name": "execute_architect_tasks",
                "description": "Execute tasks from architect's task breakdown sequentially",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "enum": ["sequential", "parallel_independent"],
                            "description": "Execute tasks in order or parallelize independent tasks"
                        }
                    }
                }
            }
        }
    ]

def _execute_architect_tasks(self, strategy="sequential"):
    """Load task-breakdown.json and delegate tasks one by one."""

    # Load task breakdown
    task_file = self.workspace_manager.workspace_path / "architecture/task-breakdown.json"
    if not task_file.exists():
        return {"error": "No task breakdown found"}

    with open(task_file) as f:
        task_data = json.load(f)

    tasks = task_data['tasks']

    # Execute tasks based on strategy
    if strategy == "sequential":
        for task in sorted(tasks, key=lambda t: t['priority']):
            if task['status'] != 'pending':
                continue

            # Check dependencies
            if not self._dependencies_met(task, tasks):
                continue

            # Delegate to task_executor
            result = self._delegate_task(task)

            # Update task status
            self._update_task_status(task_file, task['task_id'], result)

    return {"completed": sum(1 for t in tasks if t['status'] == 'completed')}
```

**Orchestrator Prompt Enhancement:**
```yaml
# orchestrator_config.yaml
system_prompt: |
  ...

  TASK EXECUTION WORKFLOW:
  1. If architect created task-breakdown.json, use execute_architect_tasks tool
  2. This will automatically distribute tasks to executor based on dependencies
  3. Tasks are executed in priority order, respecting dependencies
```

### Pros
- ✅ Explicit task-by-task execution
- ✅ Dependency-aware (only runs tasks when deps are met)
- ✅ Status tracking (updates task-breakdown.json)
- ✅ Orchestrator controls the flow
- ✅ Can parallelize independent tasks

### Cons
- ❌ Orchestrator must understand task structure
- ❌ Tight coupling between orchestrator and architect format
- ❌ More complex delegation logic
- ❌ Doesn't help with module spec awareness

### Effort
- High: ~400 lines of code, extensive testing for dependencies/parallelism
- 4-6 hours to implement and test

---

## Proposal 3: System Prompt Enhancement (QUICK WIN)

**Approach:** Add explicit instructions to task_executor prompt to check for and use architect artifacts.

### Implementation

```yaml
# task_executor_config.yaml
system_prompt: |
  You are a local coding agent that helps build software projects.

  ARCHITECTURE COORDINATION (NEW):
  **ALWAYS check for architect artifacts at startup**:
  1. Run: list_dir("architecture") to check for architecture directory
  2. If exists, read: architecture/task-breakdown.json for task list
  3. If exists, read: architecture/system-overview.md for high-level design
  4. Before implementing a module, read: architecture/modules/{module-name}.md
  5. Follow the architect's design - don't redesign from scratch

  When you complete a task from task-breakdown.json:
  - Update the task status in the JSON file
  - Add notes about what you did
  - Mark completed_at timestamp

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - Read architect artifacts FIRST before starting implementation
  ...
```

### Pros
- ✅ Zero code changes
- ✅ Immediate implementation (5 minutes)
- ✅ Can iterate based on results
- ✅ No new behaviors needed

### Cons
- ❌ Relies on LLM following instructions (not guaranteed)
- ❌ May still ignore artifacts
- ❌ No automatic injection
- ❌ Fragile - prompt changes can break it

### Effort
- Minimal: Just prompt changes
- 5 minutes to implement, testing shows if it works

---

## Proposal 4: Task Execution Tool (STRUCTURED)

**Approach:** Give task_executor a new tool to load and work through architect tasks.

### Implementation

```python
# behaviors/architect_task_execution.py
class ArchitectTaskExecutionBehavior(AgentBehavior):
    """Provides tools to load and execute architect tasks."""

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_architect_tasks",
                    "description": "Load task breakdown from architect. Returns list of tasks with dependencies, descriptions, and module specs.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_next_task",
                    "description": "Get the next pending task that has all dependencies met",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_complete",
                    "description": "Mark a task as completed in task-breakdown.json",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "result": {"type": "string", "description": "What you accomplished"}
                        },
                        "required": ["task_id", "result"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_module_spec",
                    "description": "Read module specification from architect",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {"type": "string"}
                        },
                        "required": ["module_name"]
                    }
                }
            }
        ]

    def dispatch_tool(self, tool_name, args, **kwargs):
        workspace_manager = kwargs.get('workspace_manager')

        if tool_name == "load_architect_tasks":
            return self._load_tasks(workspace_manager)

        elif tool_name == "get_next_task":
            return self._get_next_task(workspace_manager)

        elif tool_name == "mark_task_complete":
            return self._mark_complete(workspace_manager, args['task_id'], args['result'])

        elif tool_name == "read_module_spec":
            return self._read_module_spec(workspace_manager, args['module_name'])
```

**Prompt Enhancement:**
```yaml
system_prompt: |
  ...

  WORKFLOW WITH ARCHITECT:
  1. Call load_architect_tasks() at startup to see if architect provided a plan
  2. If tasks exist:
     a. Call get_next_task() to get the next actionable task
     b. Call read_module_spec(module) to read design before implementing
     c. Implement the task using write_file, run_bash, etc.
     d. Call mark_task_complete(task_id, result) when done
     e. Repeat from step 2a
  3. If no tasks exist, implement based on your goal
```

### Pros
- ✅ Explicit, structured API for task coordination
- ✅ get_next_task() handles dependency logic
- ✅ Automatic status tracking
- ✅ Clear workflow for LLM to follow
- ✅ read_module_spec() encourages reading architecture

### Cons
- ❌ Requires LLM to call tools in correct sequence
- ❌ More tools = more complexity for LLM
- ❌ New behavior to implement and maintain

### Effort
- Medium-High: ~300 lines of code, unit tests
- 3-4 hours to implement and test

---

## Proposal 5: Hybrid Approach (BALANCED)

**Approach:** Combine automatic artifact injection (Proposal 1) with explicit tools (Proposal 4) and prompt enhancement (Proposal 3).

### Implementation

1. **Add ArchitectArtifactsBehavior** - Auto-injects task summary into context
2. **Add task execution tools** - load_architect_tasks, get_next_task, mark_task_complete
3. **Enhance system prompt** - Clear instructions to use the tools

This gives task_executor:
- Automatic awareness (context injection)
- Explicit tools for structured workflow
- Clear instructions on how to use them

### Pros
- ✅ Redundant - multiple ways to discover artifacts
- ✅ Flexible - LLM can choose workflow
- ✅ Reliable - context injection ensures awareness
- ✅ Structured - tools provide clear API

### Cons
- ❌ More complexity - multiple systems for same goal
- ❌ Higher implementation effort
- ❌ More maintenance surface area

### Effort
- High: Combine Proposal 1 + 4 + 3
- 5-7 hours to implement and test

---

## Comparison Matrix

| Proposal | Effort | Reliability | Flexibility | Maintenance | Works Now? |
|----------|--------|-------------|-------------|-------------|------------|
| 1. Artifact Behavior | Medium | High | Medium | Low | After impl |
| 2. Orchestrator Control | High | High | Low | Medium | After impl |
| 3. Prompt Only | Minimal | Low | High | Minimal | Immediate |
| 4. Task Tools | Medium-High | Medium | Medium | Medium | After impl |
| 5. Hybrid | High | Very High | High | High | After impl |

---

## Recommendation

**Phase 1 (Immediate - 5 min):** Implement **Proposal 3 (Prompt Enhancement)**
- Quick win to test if prompt-based coordination works
- No code changes, immediate testing
- If LLM follows instructions, we're done!

**Phase 2 (If Proposal 3 fails - 2-3 hours):** Implement **Proposal 1 (Artifact Behavior)**
- Automatic, reliable, low maintenance
- Composable with existing behavior system
- Minimal context overhead

**Phase 3 (Future enhancement - 3-4 hours):** Add **Proposal 4 (Task Tools)**
- Structured API for explicit task management
- Better status tracking and dependency handling
- Enables orchestrator to query task progress

**Do NOT implement:** Proposal 2 (Orchestrator Control)
- Too complex
- Tight coupling
- Orchestrator shouldn't understand task structure

**Do NOT implement:** Proposal 5 (Hybrid) initially
- Overkill for first iteration
- Add tools later if needed

---

## Testing Strategy

For each proposal, test with:

1. **Simple task without architect** - Should work normally
2. **Complex task with architect** - Should read and follow artifacts
3. **Microservices task** - Should execute tasks in dependency order
4. **Mid-execution resume** - Should pick up from partially completed task breakdown

Success criteria:
- ✅ Task executor calls read_file on task-breakdown.json
- ✅ Task executor calls read_file on relevant module specs
- ✅ Implementation follows architect's design
- ✅ Task status updates in task-breakdown.json
- ✅ No redundant architecture work by executor

---

## Implementation Checklist

### Proposal 3 (Immediate)
- [ ] Update task_executor_config.yaml system prompt
- [ ] Test with L5 task (Flask API)
- [ ] Test with L7 task (microservices)
- [ ] Check logs for read_file calls on artifacts
- [ ] If successful: DONE!
- [ ] If not: Proceed to Proposal 1

### Proposal 1 (Fallback)
- [ ] Create behaviors/architect_artifacts.py
- [ ] Implement _load_artifacts() method
- [ ] Implement enhance_context() injection
- [ ] Add unit tests
- [ ] Add to task_executor_config.yaml
- [ ] Test with same L5/L7 tasks
- [ ] Verify artifacts in context via debug logs

### Proposal 4 (Enhancement)
- [ ] Create behaviors/architect_task_execution.py
- [ ] Implement load_architect_tasks tool
- [ ] Implement get_next_task tool (dependency logic)
- [ ] Implement mark_task_complete tool
- [ ] Implement read_module_spec tool
- [ ] Add unit tests for dependency resolution
- [ ] Update task_executor prompt with tool workflow
- [ ] Integration test with multi-task breakdown
