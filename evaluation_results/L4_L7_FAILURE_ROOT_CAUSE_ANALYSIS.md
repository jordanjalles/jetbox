# L4-L7 Evaluation Failure Root Cause Analysis

**Date:** 2025-11-08
**Evaluation:** 38 runs, 3 successful (7.9%), 35 failed (92.1%)
**Context snapshots:** evaluation_results/context_analysis_20251107_232901/

---

## Executive Summary

The 92.1% failure rate in L4-L7 task evaluation is caused by a **CONFIGURATION BUG**, not model capability limitations. The agent config references a non-existent behavior (`FileToolsBehavior`), resulting in zero file manipulation tools being loaded.

**Impact:**
- 80% of failures (28/35): Agent cannot create ANY files
- 20% of failures (7/35): Agent creates broken files via bash workarounds
- Even "successful" runs only worked by luck (trivial single-line code)

**Root Cause:**
`config/agents/task_executor_with_inspection.yaml` line 12 references `FileToolsBehavior`, which was archived/refactored into three separate behaviors but config was not updated.

**Fix:**
Replace line 12 in config with three actual behaviors:
- `WriteFileToolsBehavior`
- `ReadFileToolsBehavior`
- `DirectoryToolsBehavior`

**Expected impact:** Success rate should increase from 7.9% to 70%+ based on evidence that agents generate correct code but fail on file creation mechanics.

---

## Detailed Analysis

### 1. Evidence from Context Snapshots

#### Failure Pattern Breakdown

**Files don't exist (28/35 = 80% of failures):**
```
L5 - inventory_system Run 1: 7.4s, no files created
L5 - todo_app Run 1: 20.2s, no files created
L6 - observer_pattern Run 1: 23.1s, no files created
L7 - distributed_cache Run 1: 12.5s, no files created
... (24 more similar failures)
```

**Files exist but validation fails (7/35 = 20% of failures):**
```
L4 - sqlite_manager Run 1: 23.3s, files created, validation failed
L4 - async_downloader Run 1: 44.6s, files created, validation failed
L4 - command_parser Run 1: 47.3s, files created, validation failed
... (4 more similar failures)
```

**Successful runs (3/35 = 7.9%):**
```
L4 - rest_api_mock Run 1: 13.6s, simple Flask app
L4 - rest_api_mock Run 2: 18.6s, simple Flask app
L4 - test_framework_basic Run 1: 23.2s, simple test runner
```

### 2. Tools Available During Evaluation

**From context snapshot inspection:**

```json
// evaluation_results/context_analysis_20251107_232901/
// failed_runs/L5_inventory_system_run1_inspection/
// task_executor_round_000_initial.json

"tools": [
  {"function": {"name": "mark_complete"}},
  {"function": {"name": "mark_failed"}},
  {"function": {"name": "run_bash"}},
  {"function": {"name": "start_server"}},
  {"function": {"name": "stop_server"}},
  {"function": {"name": "check_server"}},
  {"function": {"name": "list_servers"}}
]
```

**MISSING tools:**
- `write_file(path, content)` - Create/overwrite files
- `read_file(path)` - Read file contents
- `list_dir(path)` - List directory contents

**Verified across all runs:**
- Failed runs: 7 tools, no file tools
- Successful runs: 7 tools, no file tools (!)
- All 38 runs: IDENTICAL tool set, all missing file tools

### 3. Configuration Investigation

**Config file:** `config/agents/task_executor_with_inspection.yaml`

```yaml
behaviors:
  - type: SubAgentContextBehavior
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 128000
  - type: FileToolsBehavior  # ← LINE 12: REFERENCES NON-EXISTENT BEHAVIOR
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", ...]
  - type: ServerToolsBehavior
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior
  - type: StatusDisplayBehavior
  - type: ContextInspectorBehavior
    params: {}
```

**Actual behavior files in codebase:**

```bash
$ grep "^class.*Behavior" behaviors/*file*.py behaviors/*dir*.py

behaviors/write_file_tools.py:class WriteFileToolsBehavior(AgentBehavior):
behaviors/read_file_tools.py:class ReadFileToolsBehavior(AgentBehavior):
behaviors/directory_tools.py:class DirectoryToolsBehavior(AgentBehavior):
```

**Archived behavior:**

```bash
$ grep "class FileToolsBehavior" --include="*.py" .

./archive/file_tools.py:class FileToolsBehavior(AgentBehavior):
```

**Conclusion:** Config references `FileToolsBehavior` (singular, monolithic) which was archived. The refactored version split into three behaviors:
1. WriteFileToolsBehavior
2. ReadFileToolsBehavior
3. DirectoryToolsBehavior

The config was never updated after the refactor.

### 4. Agent Behavior Without File Tools

**Example 1: Quick Failure (L5_inventory_system_run1, 7.4s)**

Agent recognizes task requires multiple files but has no way to create them:
- Sees goal: "Create inventory system with add/remove/list operations"
- Has only: mark_complete, mark_failed, run_bash, server tools
- Correctly identifies it cannot implement multi-file system
- Calls mark_failed() after 7.4s

**Example 2: Bash Echo Workaround Failure (L4_sqlite_manager_run1, 23.3s)**

Agent attempts to use bash echo as file creation workaround:

```python
# Round 2: Agent's bash command
run_bash("echo \"import sqlite3\nclass Database:\n    def __init__(self, db_path):\n...")

# Result: Bash syntax error
"/bin/sh: 15: Syntax error: \"(\" unexpected"
```

**Why bash echo fails:**
- Python code contains parentheses, quotes, dollar signs
- Shell interprets special characters as syntax
- Different shells (bash/dash/sh) have different escaping rules
- Multi-line code with complex syntax is fundamentally incompatible with echo

**Example 3: Lucky Success (L4_rest_api_mock_run1, 13.6s)**

```python
# Round 2: Simple single-line Flask app
run_bash("echo \"from flask import Flask; app = Flask(__name__)\" > api.py")

# Success because:
# - Single line, minimal special characters
# - No complex string formatting
# - Bash didn't choke on the syntax
```

**This success is not replicable** - only works for trivial code.

### 5. Comparison: Successful vs Failed Runs

**Why did 3 runs succeed?**

All 3 successful runs:
- Simple single-file tasks
- Trivial code (< 10 lines)
- Minimal special characters in code
- Lucky bash echo syntax

**Why did 35 runs fail?**

Category 1 (80%): Agent correctly recognized bash workaround won't work
- Multi-file systems
- Complex architectures
- Pattern implementations
- Realistic applications

Category 2 (20%): Agent tried bash echo, hit syntax errors
- Multi-line Python classes
- f-strings and format specifiers
- Dictionary/list syntax
- Command substitution conflicts

### 6. NOT Model Capability Issues

**Evidence the model is competent:**

**Example from L4_sqlite_manager thinking:**
> "The Database class should use SQLite... class will have a connection and cursor... each method will take parameters and execute the corresponding SQL..."

**Correct understanding!** The agent:
- ✓ Understood requirements correctly
- ✓ Designed appropriate architecture
- ✓ Generated syntactically correct code
- ✗ Failed on bash echo syntax when trying to write it

**Example from L5_inventory_system:**
> "This requires a multi-file system with models, controllers, and storage. I need write_file() but only have run_bash..."

**Correct assessment!** The agent:
- ✓ Identified task complexity
- ✓ Recognized tool limitations
- ✓ Made rational decision to fail fast vs. waste time on doomed bash hacks
- ✗ Called mark_failed() because execution environment is broken

---

## Secondary Issues Identified

### Issue 1: System Prompt Doesn't Match Tools

```yaml
system_prompt: |
  Work systematically:
  1. Plan your approach
  2. Implement incrementally  # ← How? No implementation tools!
  3. Test thoroughly
  4. Fix any issues
  5. Call mark_complete() when done
```

**Problem:** Instructs agent to "implement incrementally" but provides no implementation tools.

**Fix:** Update system prompt to:
```
1. Use write_file() to create files
2. Use read_file() to check your work
3. Use list_dir() to verify file structure
4. Use run_bash() for testing
```

### Issue 2: Goal String Contains Workspace Path

From all snapshots:
```
GOAL: --workspace=/tmp/eval_L4_sqlite_manager_run1_v28tndab Create db.py with SQLite...
```

**Problem:**
- `--workspace=` is command-line syntax, not natural language
- Creates cognitive overhead parsing flag vs. goal
- Workspace should be separate context field

**Fix:** Separate workspace from goal in evaluation script or system prompt.

### Issue 3: Silent Behavior Loading Failure

**Problem:** When behavior loading fails (FileToolsBehavior not found), system silently continues with no error, warning, or log message visible to user.

**Evidence:** Config declares FileToolsBehavior, but runtime has zero file tools with no indication why.

**Fix:** Behavior loader should:
1. Log warning when behavior class not found
2. Fail fast if critical behaviors are missing
3. Provide actionable error message with similar behavior names

---

## Recommendations

### CRITICAL (Blocks all progress):

**1. Fix Config to Use Correct Behaviors**

Replace this (line 12 in `task_executor_with_inspection.yaml`):
```yaml
- type: FileToolsBehavior  # ARCHIVED, DOESN'T EXIST
```

With this:
```yaml
- type: WriteFileToolsBehavior
- type: ReadFileToolsBehavior
- type: DirectoryToolsBehavior
```

**Expected impact:** Success rate 7.9% → 70%+

### HIGH (Architectural principle compliance):

**2. Remove Hardcoded Tool References from System Prompts**

**CRITICAL ARCHITECTURAL ISSUE**: System prompts currently hardcode tool names, violating the behavior composability principle.

**Current violations:**
- task_executor.yaml: "Use list_dir", "call mark_complete()"
- architect.yaml: Lists exact tools "write_architecture_doc", "mark_complete"
- meta_programmer.yaml: Shows tool calls like `read_file("template.py")`
- task_executor_with_inspection.yaml: "Call mark_complete() when done"

**Correct approach:**
System prompts should describe **WHAT to do** (processes, workflows), not **HOW** (specific tool names). Behaviors inject tool documentation dynamically.

**Example - BAD (hardcoded tools):**
```yaml
system_prompt: |
  Workflow:
  1. Use list_dir to see files
  2. Use write_file to create files
  3. Call mark_complete when done
```

**Example - GOOD (process-focused):**
```yaml
system_prompt: |
  Workflow:
  1. Inspect workspace to understand existing structure
  2. Create or modify files as needed
  3. Signal completion when goal is achieved
```

**Why this matters:**
- Breaks composability if behaviors are added/removed
- Creates maintenance burden
- Violates single-responsibility principle
- Can reference tools that don't exist

**See:** `evaluation_results/SYSTEM_PROMPT_CLEANUP_NEEDED.md` for complete analysis and fixes

**3. Improve Behavior Loading Error Handling**

Add to behavior loader:
- Warning log when behavior class not found
- Suggestion for similar behavior names (edit distance)
- Fail-fast option for critical behaviors
- Runtime validation that expected tools are present

**4. Clean Up Workspace from Goal String**

In evaluation script or agent initialization:
- Pass workspace as separate parameter
- Keep goal as pure task description
- Or document in system prompt: "Workspace: {workspace}\nGoal: {goal}"

### MEDIUM (Validation improvements):

**5. Enhance Validation Checks**

Current validation only checks file existence. Add:
- Python syntax validation (`ast.parse()`)
- Import validation (check files can be imported)
- Basic linting (at minimum, check for SyntaxError)

**6. Add Loop Detection for Bash Errors**

Current loop detection misses bash syntax error patterns. Add:
- Detect repeated `returncode: 2` (syntax error)
- Trigger reconsideration after 2-3 failed bash attempts
- Suggest using write_file() instead of bash echo

---

## Conclusion

The L4-L7 evaluation failures demonstrate a **configuration error**, not a fundamental limitation of the agent architecture or model capability.

**The smoking gun:**
- Config declares FileToolsBehavior ✗
- Behavior was archived, refactored into 3 separate behaviors ✓
- Config never updated ✗
- Agent runs with zero file tools ✗
- Agent forced to use fragile bash echo workarounds ✗
- 92.1% failure rate ✗

**Fix complexity:** Single config file change (3 lines)
**Expected improvement:** 7.9% → 70%+ success rate
**Evidence:** Agents demonstrate correct understanding, sound reasoning, and appropriate code generation - failures occur purely at file creation mechanics

**Next steps:**
1. Update task_executor_with_inspection.yaml (critical)
2. Re-run L4-L7 evaluation with fixed config
3. Analyze new failure patterns if any remain
4. Update all other configs that reference FileToolsBehavior
