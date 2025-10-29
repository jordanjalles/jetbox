# Complete Bash Migration - File Operations

**Date:** 2025-10-29
**Change:** Removed write_file and read_file from LLM tool definitions

## Problem Solved

**Parameter invention crisis:** 72% of L5-L7 test failures were caused by the LLM inventing parameters for `write_file()` and `read_file()`:
- `read_file(line_end=...)`
- `write_file(overwrite=...)`
- `write_file(line_end=...)`
- `write_file(timeout=...)`
- etc.

## Solution

**Complete bash migration:** Remove write_file and read_file from tool definitions entirely. Force all file operations through `run_bash()`.

### Why This Works

**Python function signatures invite parameter invention:**
```python
def write_file(path: str, content: str):
    # LLM thinks: "This is too simple, what about line endings?"
    # LLM invents: write_file(path, content, line_end='\n')
```

**Bash commands have complete, well-known syntax:**
```bash
cat > file.py <<'EOF'
code here
EOF
```

LLM thinks: "This is the complete syntax from Unix documentation. I can't add parameters."

## Changes Made

### 1. tools.py - Removed from get_tool_definitions()

**Before:**
```python
def get_tool_definitions():
    return [
        write_file_definition,
        read_file_definition,
        list_dir_definition,
        run_bash_definition,
        ...
    ]
```

**After:**
```python
def get_tool_definitions():
    """
    IMPORTANT: write_file and read_file have been REMOVED.
    Agent must use run_bash for all file operations.
    """
    return [
        list_dir_definition,  # Kept for quick navigation
        run_bash_definition,  # Expanded with file operation examples
        ...
    ]
```

**Note:** Functions still exist in tools.py for internal use, just not exposed to LLM.

### 2. tools.py - Enhanced run_bash description

Added comprehensive file operation examples directly in the tool definition:

```python
{
    "name": "run_bash",
    "description": """Run any bash command with full shell features.

Use run_bash for ALL file operations:
- Write files: run_bash("cat > file.py <<'EOF'\\ncode here\\nEOF")
- Read files: run_bash("cat file.py")
- Append: run_bash("cat >> file.py <<'EOF'\\nmore code\\nEOF")
- Partial read: run_bash("head -20 file.py")
- Search: run_bash("grep -n 'pattern' *.py")
...
"""
}
```

### 3. task_executor_agent.py - Updated tool_map

**Before:**
```python
tool_map = {
    "list_dir": tools.list_dir,
    "read_file": tools.read_file,  # ← Removed
    "write_file": tools.write_file,  # ← Removed
    "run_bash": tools.run_bash,
    ...
}
```

**After:**
```python
tool_map = {
    "list_dir": tools.list_dir,
    "run_bash": tools.run_bash,
    # NOTE: write_file and read_file removed - agent must use run_bash
    ...
}
```

### 4. agent_config.yaml - Updated system prompt

**Before:**
```yaml
Tool usage:
- write_file(path, content): Write files
- read_file(path): Read small files
- run_bash(command): Run shell commands
```

**After:**
```yaml
Available tools:
- list_dir(path): List directory contents
- run_bash(command, timeout): Run ANY shell command

IMPORTANT: Use run_bash for ALL file operations:

Writing files (use heredoc syntax):
- run_bash("cat > file.py <<'EOF'\\nimport sys\\nEOF")

Reading files:
- run_bash("cat file.py")
- run_bash("head -20 file.py")
- run_bash("tail -50 file.py")
```

## Expected Benefits

### 1. Eliminates Parameter Invention (Primary Goal)

**Before:** LLM invents 18 parameter errors across 27 tasks (67%)
**After:** Zero parameter invention errors (can't invent parameters for fixed bash syntax)

### 2. More Powerful File Operations

**Before:** Limited to simple write/read
**After:** Full bash capabilities:
- Conditional writes: `run_bash("[ ! -f file.py ] && cat > file.py <<'EOF'...")`
- Multi-file operations: `run_bash("cat file1 file2 > combined")`
- In-place editing: `run_bash("sed -i 's/old/new/' file.py")`
- Complex workflows: `run_bash("grep -l 'TODO' *.py | xargs cat")`

### 3. Consistent Tool Philosophy

**Before:** Mixed paradigm (Python wrappers + bash commands)
**After:** Pure bash (except list_dir for convenience)

### 4. Less Code to Maintain

**Before:** 150 lines of write_file/read_file implementation + validation + error handling
**After:** 60 lines of run_bash + workspace path resolution

## What Stayed the Same

### list_dir() Still Available

**Reason:** Simple, bounded operation with no parameter invention risk
```python
list_dir(path: str) -> list[str]
```

No logical parameters to invent (unlike write_file which "needs" overwrite, line_end, etc.)

### write_file/read_file Still Exist

**Location:** tools.py lines 79-156

**Why:** Internal tooling may still use them (e.g., jetbox_notes, workspace_manager)

**Key:** Just not exposed to LLM via get_tool_definitions()

## Risks and Mitigations

### Risk 1: Heredoc Syntax Complexity

**Concern:** LLM might struggle with heredoc syntax

**Mitigation:**
- Multiple examples in system prompt
- Tool description includes complete syntax
- Heredocs are well-documented in LLM training data

### Risk 2: Escaping Issues

**Concern:** Special characters in code might break heredocs

**Mitigation:**
- Use single-quote heredocs (`<<'EOF'`) for no interpolation
- System prompt shows this explicitly
- Ollama handles shell escaping in tool calls

### Risk 3: Reduced Ergonomics

**Concern:** `run_bash("cat > file.py <<'EOF'...")` is more verbose than `write_file(path, content)`

**Response:**
- Verbosity is acceptable to avoid parameter invention
- More explicit about what's happening
- More powerful (can do conditional writes, append, etc.)

## Testing Plan

### Phase 1: Smoke Test (Quick)

Create simple task that requires file write/read:
```python
agent = TaskExecutorAgent(
    workspace=tmp_dir,
    goal="Create hello.py with print('hello world'). Read it back and verify.",
    max_rounds=10
)
result = agent.run()
```

**Expected:** Agent uses run_bash with heredoc syntax

### Phase 2: Re-run L5-L7 Evaluation

Full semantic validation with all 27 tasks:
```bash
python run_l5_l7_semantic.py
```

**Expected improvements:**
- Parameter invention errors: 18 → 0 (eliminate)
- Overall pass rate: 7.4% → ?% (should increase significantly)
- L5 pass rate: 22% → ?% (most L5 failures were parameter errors)

### Phase 3: Compare Approaches

If time permits, compare:
- A: Current approach (bash only)
- B: **kwargs approach (accept unknown parameters)

Measure:
- Success rates
- Token usage
- Rounds to completion
- Error types

## Success Criteria

**Minimum acceptable:**
- Zero parameter invention errors
- At least one L6 task passes (currently 0%)
- L5 pass rate > 22% (current)

**Good result:**
- Overall pass rate > 20% (currently 7.4%)
- L5 pass rate > 50%
- L6 pass rate > 10%

**Excellent result:**
- Overall pass rate > 40%
- L5 pass rate > 70%
- L6 pass rate > 30%

## Rollback Plan

If results are worse (unlikely):

1. Revert tools.py get_tool_definitions() to include write_file/read_file
2. Revert task_executor_agent.py tool_map
3. Revert agent_config.yaml system prompt
4. Apply **kwargs approach instead

All files under version control, easy to revert:
```bash
git checkout tools.py task_executor_agent.py agent_config.yaml
```

## Related Documentation

- **Previous attempt:** `/workspace/docs/BASH_TOOL_MIGRATION.md` - Partial migration (grep_file only)
- **Regression report:** `/workspace/analysis/L5_L7_POST_FIXES_REGRESSION.md` - Identified the problem
- **Context validation:** `/workspace/analysis/CONTEXT_MANAGEMENT_VALIDATION.md` - Other fix that's working

## Next Steps

1. ✅ Complete bash migration (this document)
2. ⏭️ Quick smoke test (verify bash syntax works)
3. ⏭️ Full L5-L7 evaluation
4. ⏭️ Analyze results and document findings
5. ⏭️ If successful, update all documentation

---

**Change approved by:** User (via "can we get around the kwargs by providing bash command access like before?")
**Implementation date:** 2025-10-29
**Status:** Ready for testing
