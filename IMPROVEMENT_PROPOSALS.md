# Jetbox Improvement Proposals

**Date:** 2025-11-04
**Status:** Draft
**Author:** Code Inspection & Testing Analysis

## Executive Summary

Based on comprehensive code inspection (44 issues found) and edge case testing (8 tests), this document proposes improvements across 5 categories:

1. **Critical Fixes** (3 proposals) - Bugs found during testing
2. **High-Priority Refactoring** (6 proposals) - Code duplication and architecture issues
3. **Medium-Priority Enhancements** (8 proposals) - Usability and robustness
4. **Low-Priority Cleanup** (4 proposals) - Code quality improvements
5. **Future Enhancements** (3 proposals) - New capabilities

**Total:** 24 improvement proposals

---

## 1. Critical Fixes (Must Fix)

### CRITICAL-1: Empty Goal Handling

**Issue Found:** Edge Case Test 1
**Current Behavior:** `python orchestrator_agent.py --once ""` enters infinite loop
**Expected Behavior:** Should reject empty goal with error message

**Root Cause:**
- `BaseAgent.main()` doesn't validate empty goals before entering execution
- Orchestrator falls back to interactive mode when goal is empty with --once

**Proposed Fix:**
```python
# In BaseAgent.main()
if args.get('initial_message') == "":
    print("Error: Goal cannot be empty")
    print(f"Usage: python {sys.argv[0]} 'goal description'")
    sys.exit(1)
```

**Priority:** CRITICAL
**Effort:** 1 hour
**Risk:** Low

---

### CRITICAL-2: EOF Error Handling in Interactive Mode

**Issue Found:** Test 1 - EOF when reading input
**Current Behavior:** Unhandled EOFError causes traceback spam in fallback loop
**Expected Behavior:** Graceful exit on EOF (Ctrl+D)

**Root Cause:**
- `orchestrator_agent.py:218` - `input()` not wrapped in try/except for EOF
- Fallback loop doesn't handle stdin closure

**Proposed Fix:**
```python
# In BaseAgent.run_agent() fallback loop
try:
    user_input = input("You: ").strip()
except (EOFError, KeyboardInterrupt):
    print("\nShutting down...")
    break
```

**Priority:** CRITICAL
**Effort:** 30 minutes
**Risk:** Low

---

### CRITICAL-3: Subprocess Timeout Handling

**Issue:** delegation.py subprocess calls have 600s timeout but no recovery logic
**Current Behavior:** On timeout, returns generic error
**Expected Behavior:** Should provide actionable information about timeout

**Root Cause:**
- `behaviors/delegation.py:328-334` catches TimeoutExpired but message is not helpful
- No indication of which agent timed out or why

**Proposed Fix:**
```python
except subprocess.TimeoutExpired:
    return {
        "success": False,
        "message": f"Task execution timed out after 600 seconds. The {target_agent_name} agent may be stuck in a loop or the task is too complex. Consider breaking it into smaller subtasks.",
        "timeout": True
    }
```

**Priority:** HIGH
**Effort:** 1 hour
**Risk:** Low

---

## 2. High-Priority Refactoring

### REFACTOR-1: Remove Duplicated Completion Tools

**Issue:** Code Inspection Issue 1.1 (High Severity)
**Current State:** 3 behaviors define near-identical completion tools:
- `behaviors/subagent_mode.py:144-179` (mark_complete, mark_failed)
- `behaviors/compact_when_near_full.py:247-272` (mark_goal_complete)

**Problem:**
- Code duplication (35+ lines × 2 = 70 lines)
- Inconsistent parameter names (summary vs description)
- Maintenance burden (fix bugs in 2-3 places)

**Proposed Solution:**
Create `behaviors/completion_tools.py`:
```python
class CompletionToolsBehavior(AgentBehavior):
    """Provides mark_complete and mark_failed tools."""

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_complete",
                    "description": "Mark the current goal/task as successfully completed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of what was accomplished"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            },
            # ... mark_failed tool
        ]
```

Then remove from subagent_mode and compact_when_near_full.

**Benefits:**
- Single source of truth
- Easier to maintain
- Consistent across all agents

**Priority:** HIGH
**Effort:** 4 hours
**Risk:** Medium (need to update all agent configs)

---

### REFACTOR-2: Extract LLM Summarization Utility

**Issue:** Code Inspection Issue 1.9 (Medium Severity)
**Current State:** 3 behaviors duplicate LLM summarization pattern:
- `compact_when_near_full.py:188-246`
- `workspace_task_notes.py:118-161, 163-239`

**Problem:** 120+ lines of duplicated code with identical structure

**Proposed Solution:**
Create `llm_utils.py`:
```python
def summarize_with_llm(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    timeout: int = 30,
    max_tokens: int = 500
) -> str:
    """Generic LLM summarization with error handling."""
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Summarization failed: {e}]"
```

**Benefits:**
- Reduces 120 lines to 20
- Centralized error handling
- Easier to add caching or rate limiting

**Priority:** HIGH
**Effort:** 3 hours
**Risk:** Low

---

### REFACTOR-3: Make Hardcoded Paths Configurable

**Issue:** Code Inspection Issue 3.4 (Medium Severity)
**Current State:** Hardcoded paths in delegation.py:
```python
msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
```

**Problem:**
- Assumes parent is orchestrator
- Won't work for multi-level delegation hierarchies

**Proposed Solution:**
Add to behavior config:
```yaml
# In delegation.py __init__
def __init__(self, can_delegate_to, parent_agent_name="orchestrator", ...):
    self.parent_agent_name = parent_agent_name
    self.message_file = Path(f".agent_context/messages_to_{parent_agent_name}.jsonl")
```

Or use generic: `messages_to_parent.jsonl`

**Priority:** HIGH
**Effort:** 2 hours
**Risk:** Low

---

### REFACTOR-4: Context Injection Helper Method

**Issue:** Code Inspection Issue 1.3 (Medium Severity)
**Current State:** 4+ behaviors duplicate context injection pattern

**Proposed Solution:**
Add to `AgentBehavior` base class:
```python
class AgentBehavior:
    def inject_user_message_after_system(
        self,
        context: list[dict],
        message: str
    ) -> list[dict]:
        """
        Inject a user message after system prompt.

        Standard pattern used by behaviors to inject warnings/nudges.
        """
        if len(context) > 0:
            context.insert(1, {"role": "user", "content": message})
        return context
```

**Benefits:**
- Reduces boilerplate in 4+ behaviors
- Standardizes injection pattern
- Easier to modify behavior later

**Priority:** MEDIUM-HIGH
**Effort:** 2 hours
**Risk:** Low

---

### REFACTOR-5: Remove Commented-Out Deprecated Code

**Issue:** Code Inspection Issue 2.1, 2.2 (Low Severity but widespread)
**Current State:** 15+ files with commented-out code:
- `behaviors/subagent_mode.py:353-359, 387-390`
- Many others with "DEPRECATED" comments

**Problem:**
- Clutter and confusion
- Version control already preserves history
- Makes code harder to read

**Proposed Solution:**
Remove all commented-out code blocks with DEPRECATED markers:
```bash
# Find and review all commented deprecated code
grep -r "# DEPRECATED" --include="*.py" .
```

**Benefits:**
- Cleaner codebase
- Easier to read
- Reduces maintenance burden

**Priority:** MEDIUM
**Effort:** 2 hours
**Risk:** Very Low

---

### REFACTOR-6: Standardize Token Estimation

**Issue:** Code Inspection Issue 1.4 (Medium Severity)
**Current State:** Multiple behaviors estimate tokens differently

**Proposed Solution:**
Create `context_utils.py`:
```python
def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count for message list."""
    # Rough estimate: 4 chars = 1 token
    total_chars = sum(len(str(m)) for m in messages)
    return total_chars // 4

def get_max_tokens_for_agent(agent: Any) -> int:
    """Get max context tokens from agent or behaviors."""
    # Try multiple sources
    if hasattr(agent, 'max_context_tokens'):
        return agent.max_context_tokens

    # Check behaviors for CompactWhenNearFullBehavior
    for behavior in getattr(agent, 'behaviors', []):
        if hasattr(behavior, 'max_tokens'):
            return behavior.max_tokens

    return 128000  # Default
```

**Priority:** MEDIUM
**Effort:** 3 hours
**Risk:** Low

---

## 3. Medium-Priority Enhancements

### ENHANCE-1: Workspace Path Validation

**Issue:** Edge cases with workspace paths
**Current State:** No validation of workspace_path parameter

**Proposed Enhancement:**
```python
def validate_workspace_path(path: str, mode: str) -> tuple[bool, str]:
    """
    Validate workspace path based on mode.

    Returns: (is_valid, error_message)
    """
    if mode == "existing":
        if not path:
            return False, "workspace_path required when workspace_mode='existing'"

        path_obj = Path(path)
        if not path_obj.exists():
            return False, f"Workspace does not exist: {path}"

        if not path_obj.is_dir():
            return False, f"Workspace path is not a directory: {path}"

    elif mode == "new":
        if path:
            return False, "workspace_path should not be provided when workspace_mode='new'"

    return True, ""
```

**Priority:** MEDIUM
**Effort:** 2 hours
**Risk:** Low

---

### ENHANCE-2: Better Error Messages for Tool Parameter Validation

**Issue:** Current validation messages are generic
**Current State:** "Missing required parameter X"

**Proposed Enhancement:**
```python
# Instead of:
"Missing required parameter: task_description"

# Provide:
"Missing required parameter 'task_description'.
Example: delegate_to_executor(task_description='Create a calculator', workspace_mode='new')"
```

Include example usage in error messages for better DX.

**Priority:** MEDIUM
**Effort:** 3 hours
**Risk:** Low

---

### ENHANCE-3: Structured Logging

**Issue:** Mix of print statements makes debugging hard
**Current State:**
```python
print(f"[orchestrator] Round {round_num}/100")
print(f"[delegation] Cleared OLLAMA_MODEL...")
```

**Proposed Enhancement:**
```python
import logging

logger = logging.getLogger("jetbox")
logger.setLevel(logging.INFO)

# Usage:
logger.info("[orchestrator] Round %d/100", round_num)
logger.debug("[delegation] Cleared OLLAMA_MODEL for %s", agent_name)
```

**Benefits:**
- Configurable log levels
- Better debugging
- Can redirect to files
- Timestamp support

**Priority:** MEDIUM
**Effort:** 4 hours
**Risk:** Low

---

### ENHANCE-4: Agent Health Checks

**Issue:** No way to verify agents are configured correctly
**Proposed Enhancement:**

```bash
# New command:
python agent.py --health-check

# Output:
✓ Orchestrator agent loaded successfully
✓ Found 6 behaviors: delegation, chatbot, ...
✓ Can delegate to: architect, task_executor
✓ 8 tools available: consult_architect, delegate_to_executor, ...
✓ System prompt loaded (2863 chars)
✓ Model: qwen3:8b (available via Ollama)
```

**Priority:** MEDIUM
**Effort:** 3 hours
**Risk:** Low

---

### ENHANCE-5: Delegation Result Caching

**Issue:** Re-delegating same task runs it again
**Proposed Enhancement:**

Cache delegation results based on:
- Target agent
- Task description hash
- Workspace path

```python
# In DelegationBehavior
def _get_cache_key(target_agent, task_description, workspace):
    content = f"{target_agent}::{task_description}::{workspace}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def _check_cache(cache_key):
    cache_file = Path(f".agent_context/delegation_cache/{cache_key}.json")
    if cache_file.exists():
        # Check if result is still valid (workspace unchanged)
        ...
```

**Priority:** LOW-MEDIUM
**Effort:** 6 hours
**Risk:** Medium (need invalidation logic)

---

### ENHANCE-6: Workspace Cleanup Tool

**Issue:** `.agent_workspaces/` can accumulate hundreds of directories
**Proposed Enhancement:**

```bash
# New command:
python agent.py --clean-workspaces

# Options:
--clean-workspaces           # Interactive cleanup
--clean-workspaces --older-than=7d   # Remove workspaces older than 7 days
--clean-workspaces --dry-run   # Show what would be deleted
```

**Priority:** MEDIUM
**Effort:** 4 hours
**Risk:** Low

---

### ENHANCE-7: Behavior Dependency Declarations

**Issue:** Some behaviors depend on others being present
**Current State:** No way to declare dependencies

**Proposed Enhancement:**
```python
class WorkspaceTaskNotesBehavior(AgentBehavior):
    def get_dependencies(self) -> list[str]:
        """Return list of required behavior names."""
        return ["subagent_mode"]  # Needs SubAgentModeBehavior

    def get_conflicts(self) -> list[str]:
        """Return list of conflicting behavior names."""
        return []  # No conflicts
```

Then validate in BaseAgent:
```python
def _validate_behavior_dependencies(self):
    """Check all behaviors have required dependencies."""
    for behavior in self.behaviors:
        for dep in behavior.get_dependencies():
            if dep not in [b.get_name() for b in self.behaviors]:
                raise ValueError(f"{behavior.get_name()} requires {dep} behavior")
```

**Priority:** MEDIUM
**Effort:** 3 hours
**Risk:** Low

---

### ENHANCE-8: Config File Validation

**Issue:** Typos in YAML configs cause silent failures
**Proposed Enhancement:**

```python
# New module: config_validator.py
def validate_agent_config(config_path: Path):
    """Validate agent config against schema."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check required fields
    required = ["role", "system_prompt", "behaviors"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Validate behavior types
    for behavior in config.get("behaviors", []):
        behavior_type = behavior.get("type")
        if not is_valid_behavior(behavior_type):
            raise ValueError(f"Unknown behavior type: {behavior_type}")

    return True
```

**Priority:** MEDIUM
**Effort:** 4 hours
**Risk:** Low

---

## 4. Low-Priority Cleanup

### CLEANUP-1: Remove Unused Private Methods

**Issue:** Code Inspection Issue 2.3
**Methods:** `base_agent.py:_to_snake_case`, etc.

**Proposed:** Move to shared `utils.py` or remove if truly unused

**Priority:** LOW
**Effort:** 2 hours
**Risk:** Low

---

### CLEANUP-2: Improve Empty Except Blocks

**Issue:** Code Inspection Issue 2.8
**Current:** Silent exception swallowing

**Proposed:**
```python
except Exception as e:
    logger.warning(f"[{self.name}] Failed to load state: {e}")
    # Continue with fresh state
```

**Priority:** LOW
**Effort:** 1 hour
**Risk:** Low

---

### CLEANUP-3: Simplify Action Hashing

**Issue:** Code Inspection Issue 2.7
**Current:** Complex recursive serialization

**Proposed:**
```python
def _hash_action(self, action):
    return json.dumps(action, default=str, sort_keys=True)
```

**Priority:** LOW
**Effort:** 1 hour
**Risk:** Low

---

### CLEANUP-4: Remove Global State Variable

**Issue:** Code Inspection Issue 2.5
**Location:** `workspace_task_notes.py:22-23`

**Proposed:** Pass workspace_manager through parameters

**Priority:** LOW
**Effort:** 2 hours
**Risk:** Low

---

## 5. Future Enhancements

### FUTURE-1: Parallel Delegation

**Vision:** Delegate to multiple agents concurrently

**Example:**
```python
# Instead of:
consult_architect(...)
delegate_to_executor(...)

# Enable:
parallel_delegate([
    ("architect", "Design auth system"),
    ("task_executor", "Create test fixtures"),
])
```

**Priority:** FUTURE
**Effort:** 2 weeks
**Risk:** High

---

### FUTURE-2: Agent Marketplace/Plugins

**Vision:** Third-party agents as plugins

**Example:**
```yaml
# agents.yaml
agents:
  custom_linter:
    plugin: "jetbox-linter-plugin"
    version: "1.0.0"
    can_delegate_to: []
```

**Priority:** FUTURE
**Effort:** 4 weeks
**Risk:** High

---

### FUTURE-3: Web UI Dashboard

**Vision:** Real-time monitoring of agent activity

**Features:**
- Live delegation tree visualization
- Real-time token usage
- Workspace browser
- Historical runs

**Priority:** FUTURE
**Effort:** 6 weeks
**Risk:** Medium

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
1. CRITICAL-1: Empty goal handling
2. CRITICAL-2: EOF error handling
3. CRITICAL-3: Subprocess timeout handling

**Expected Outcome:** Zero crashes on edge cases

---

### Phase 2: High-Priority Refactoring (Weeks 2-3)
1. REFACTOR-1: Completion tools consolidation
2. REFACTOR-2: LLM summarization utility
3. REFACTOR-3: Configurable paths
4. REFACTOR-4: Context injection helper
5. REFACTOR-5: Remove deprecated code
6. REFACTOR-6: Token estimation standardization

**Expected Outcome:** 200+ lines removed, cleaner architecture

---

### Phase 3: Medium-Priority Enhancements (Weeks 4-6)
1. ENHANCE-1: Workspace validation
2. ENHANCE-2: Better error messages
3. ENHANCE-3: Structured logging
4. ENHANCE-4: Health checks
5. ENHANCE-5: Result caching
6. ENHANCE-6: Workspace cleanup
7. ENHANCE-7: Behavior dependencies
8. ENHANCE-8: Config validation

**Expected Outcome:** Production-ready robustness

---

### Phase 4: Low-Priority Cleanup (Week 7)
1. CLEANUP-1: Remove unused methods
2. CLEANUP-2: Improve exception handling
3. CLEANUP-3: Simplify hashing
4. CLEANUP-4: Remove global state

**Expected Outcome:** Code quality A+

---

## Metrics & Success Criteria

### Code Quality Metrics
- **Current Lines of Code:** ~15,000
- **Target Reduction:** -500 lines (3%) via deduplication
- **Current Issues:** 44 (from inspection)
- **Target Issues:** <10 high/medium severity

### Performance Metrics
- **Delegation Latency:** Currently ~5-10s overhead
- **Target:** <3s overhead via caching
- **Token Usage:** Track and optimize (no current baseline)

### Reliability Metrics
- **Edge Case Pass Rate:** Currently 62% (5/8 tests)
- **Target:** 100% (8/8 tests)
- **Crash Rate:** Currently >0 (EOF errors)
- **Target:** 0 crashes on valid input

---

## Appendix: Test Results Summary

### Edge Case Testing (8 Tests)

✅ **PASSED (5):**
1. Task executor rejects empty goal correctly
2. Long goals (200+ words) processed successfully
3. Special characters in goals handled
4. Direct TaskExecutor invocation works
5. Workspace reuse via delegation works

⚠️ **FAILED (2):**
1. Orchestrator --once with empty goal → infinite loop
2. EOF in fallback mode → unhandled exception

⏭️ **SKIPPED (1):**
1. Nested delegation (orchestrator→executor→...) - not implemented

### Core Functionality Testing (5 Tests)

✅ **ALL PASSED:**
1. Basic orchestrator chat mode
2. Orchestrator autonomous mode
3. TaskExecutor direct invocation
4. Orchestrator delegates to TaskExecutor
5. Workspace reuse - delegation to existing workspace

---

## Conclusion

This proposal outlines 24 improvements across 5 categories. Implementing Phase 1-3 (critical fixes + refactoring + enhancements) over 6 weeks would significantly improve code quality, robustness, and maintainability.

**Recommended Starting Point:** Implement critical fixes in Phase 1 immediately (3 fixes, ~3 hours total).

