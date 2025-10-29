# Bash Migration: Heredoc JSON Parsing Error

**Date:** 2025-10-29
**Context:** Full bash migration evaluation revealed new JSON parsing errors

## Problem Discovery

After completing the bash migration (removing write_file/read_file from tool definitions), the L5-L7 evaluation revealed **16 JSON parsing errors** across 27 tasks (59% error rate from JSON issues alone).

### Error Pattern

**Typical error:**
```
error parsing tool call: raw='{"command":"cat > blog.py <<'EOF'\n...code here...\nEOF"}',
err=invalid character ']' after object key:value pair (status code: -1)
```

**Breakdown:**
- 8 errors with `<<'EOF'` delimiter
- 8 errors with `<<'PATCH'` or other quoted delimiters
- **Total: 16 JSON parsing errors / 27 tasks = 59% JSON error rate**

## Root Cause Analysis

### The Fundamental Incompatibility

**Bash heredoc syntax (correct):**
```bash
cat > file.py <<'EOF'
code here
EOF
```

**Why the quotes are needed:** The `<<'EOF'` syntax (with quotes) disables variable expansion. Without quotes, bash would expand variables like `$VAR` inside the heredoc, which is dangerous when writing code.

**Ollama JSON tool calling format:**
```json
{
  "command": "cat > file.py <<'EOF'\ncode\nEOF"
}
```

**The problem:** JSON strings cannot contain unescaped single quotes. When the LLM generates `<<'EOF'` inside a JSON string value, it breaks JSON parsing:

```
"cat > file.py <<'EOF'\n..."
                 ^--- This single quote breaks JSON parsing
```

### Why This Happens

1. LLM correctly learns bash heredoc syntax requires `<<'EOF'`
2. LLM generates tool call with heredoc syntax
3. Ollama serializes tool call to JSON
4. JSON parser fails on unescaped single quotes
5. Ollama returns status code -1 (JSON parse error)
6. Agent crashes

## Attempted Solutions

### Attempt 1: Use heredoc without quotes (`<<EOF`)

**Format:**
```bash
cat > file.py <<EOF
code here
EOF
```

**Problem:** Without quotes, bash expands variables. If the code contains `$variable`, bash will try to substitute it:

```python
# Python code to write:
price = $10  # Bash sees $10 and tries to expand it!
```

**Result:** Data corruption, unexpected behavior, security risk.

### Attempt 2: Use different delimiter

**Format:**
```bash
cat > file.py <<ENDOFFILE
code here
ENDOFFILE
```

**Problem:** Still needs quotes for safety: `<<'ENDOFFILE'`. Same JSON issue, just different delimiter name.

### Attempt 3: Use printf/echo

**Format:**
```bash
printf '%s\n' 'line1' 'line2' 'line3' > file.py
```

**Problems:**
- Extremely verbose for multi-line files
- Hard to generate (LLM must quote every line)
- Error-prone (missing quotes, escaping issues)
- Token-expensive

### Attempt 4: Base64 encode

**Format:**
```bash
echo 'base64encodedcontent' | base64 -d > file.py
```

**Problems:**
- Not human-readable
- Adds complexity for no benefit
- LLM would struggle to generate base64

## Comparison: Parameter Invention vs JSON Errors

### Previous approach (write_file with parameter invention)
- **Parameter invention errors:** 18/27 tasks (67%)
- **JSON parsing errors:** 0/27 tasks (0%)
- **Total failure rate:** 67%

### Current approach (bash heredoc)
- **Parameter invention errors:** 0/27 tasks (0%) ✓ Eliminated!
- **JSON parsing errors:** 16/27 tasks (59%) ✗ New problem!
- **Total failure rate:** 59%

**Marginal improvement:** 67% → 59% (8% reduction)
**Trade-off:** Traded one error type for another

## The Real Solution: **kwargs

Instead of fighting bash/JSON incompatibility, accept and ignore invented parameters:

```python
def write_file(path: str, content: str, **kwargs) -> str:
    """
    Write/overwrite a text file.

    Args:
        path: File path
        content: File contents
        **kwargs: Ignored (allows LLM to pass any parameters harmlessly)
    """
    if kwargs:
        print(f"[tools] Ignoring unexpected write_file params: {list(kwargs.keys())}")

    # ... existing implementation ...
```

### Why This Works

**Eliminates parameter invention crashes:**
- LLM invents `line_end='\\n'` → ignored, no error
- LLM invents `overwrite=True` → ignored, no error
- LLM invents `timeout=60` → ignored, no error

**Eliminates JSON parsing errors:**
- No need for heredoc syntax
- LLM uses simple `write_file(path, content)`
- JSON is straightforward: `{"path": "file.py", "content": "..."}`

**Clean fallback behavior:**
- Logs unexpected parameters (helpful for debugging)
- Continues execution (no crash)
- Uses sensible defaults (overwrite=True, UTF-8 encoding, etc.)

### Comparison: Bash Migration vs **kwargs

| Metric | Bash Migration | **kwargs Approach |
|--------|----------------|-------------------|
| Parameter invention errors | 0% ✓ | 0% ✓ |
| JSON parsing errors | 59% ✗ | 0% ✓ |
| Ergonomics | Poor (heredoc verbose) | Good (simple function call) |
| Code clarity | Mixed (bash + tool calls) | Clean (consistent tool calls) |
| Maintenance | High (bash edge cases) | Low (Python only) |
| Token usage | High (heredoc boilerplate) | Low (concise calls) |
| Human readability | Poor (heredoc syntax) | Good (readable JSON) |

**Clear winner:** **kwargs approach

## Implementation: Revert to write_file with **kwargs

### Step 1: Update tools.py

```python
def write_file(path: str, content: str, **kwargs) -> str:
    """Write/overwrite a text file. Ignores unexpected parameters."""
    if kwargs:
        ignored = ", ".join(kwargs.keys())
        print(f"[tools] write_file ignoring parameters: {ignored}")
    # ... existing implementation ...

def read_file(path: str, **kwargs) -> str:
    """Read a text file. Ignores unexpected parameters."""
    if kwargs:
        ignored = ", ".join(kwargs.keys())
        print(f"[tools] read_file ignoring parameters: {ignored}")
    # ... existing implementation ...
```

### Step 2: Update get_tool_definitions()

Add write_file and read_file back to tool definitions:
```python
{
    "name": "write_file",
    "description": "Write/overwrite a text file",
    "parameters": {
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["path", "content"]
    }
}
```

No need to list every possible parameter - **kwargs handles them all.

### Step 3: Update task_executor_agent.py

Add back to tool_map:
```python
tool_map = {
    "write_file": tools.write_file,
    "read_file": tools.read_file,
    "list_dir": tools.list_dir,
    "run_bash": tools.run_bash,
    ...
}
```

### Step 4: Update agent_config.yaml

Simplify system prompt:
```yaml
Available tools:
- write_file(path, content): Write files
- read_file(path): Read files
- list_dir(path): List directory
- run_bash(command): Run shell commands
```

Keep run_bash for complex operations, but use write_file for simple writes.

## Benefits of **kwargs Approach

### 1. Best of Both Worlds

- ✓ Eliminates parameter invention crashes (like bash migration)
- ✓ Eliminates JSON parsing errors (unlike bash migration)
- ✓ Simple ergonomics (unlike bash heredocs)
- ✓ Graceful degradation (logs ignored params)

### 2. Aligns with Python Philosophy

**"Easier to ask forgiveness than permission" (EAFP)**

Instead of:
- Restricting LLM to exact parameters (fails on invention)
- Forcing LLM to use bash syntax (fails on JSON)

We accept whatever the LLM passes and use what we need.

### 3. Future-Proof

If the LLM invents new parameters in the future:
- No code changes needed
- No crashes
- Logged for analysis
- Easy to promote to real parameters if useful

### 4. Maintains Workspace Safety

All existing safety features remain:
- Workspace path resolution
- Edit mode restrictions
- File tracking
- Ledger logging

## Performance Impact

### Token Usage

**Bash heredoc approach:**
```json
{
  "command": "cat > models.py <<'EOF'\nimport sys\n\nclass Model:\n    pass\nEOF"
}
```
~40 tokens overhead per write

**write_file approach:**
```json
{
  "path": "models.py",
  "content": "import sys\n\nclass Model:\n    pass"
}
```
~10 tokens overhead per write

**Savings:** ~30 tokens per file write (~75% reduction in overhead)

### Error Recovery

**Bash heredoc:** JSON error → immediate crash → Ollama context clear → retry from start

**write_file + **kwargs:** Ignored parameter → warning logged → continue → no retry needed

**Time savings:** Eliminates crash/retry cycle (saves 30-60s per occurrence)

## Recommendation

**Revert the bash migration** and implement **kwargs approach instead.

This is not a failure of the bash migration concept - it correctly identified and solved the parameter invention problem. However, it introduced a worse problem (JSON parsing errors) that we didn't anticipate.

The **kwargs solution is superior because it:
1. Solves both problems (parameter invention AND JSON errors)
2. Simpler to implement
3. Better ergonomics
4. More maintainable
5. More performant

## Testing Plan

### Phase 1: Implementation (10 minutes)
1. Add **kwargs to write_file/read_file
2. Re-add to get_tool_definitions()
3. Re-add to task_executor_agent.py tool_map
4. Update agent_config.yaml

### Phase 2: Smoke Test (5 minutes)
Run test_bash_file_ops.py (should work identically)

### Phase 3: Full Evaluation (20 minutes)
Run L5-L7 evaluation, expect:
- Parameter invention errors: 0 (ignored by **kwargs)
- JSON parsing errors: 0 (no heredocs)
- Overall pass rate: > 7.4% (current baseline)

### Phase 4: Comparison Report
Document:
- Error reduction
- Pass rate improvement
- Token usage comparison
- Performance metrics

## Lessons Learned

1. **Test edge cases before full migration** - Should have tested heredoc JSON compatibility first

2. **Simple solutions often better** - **kwargs is simpler than bash migration

3. **Don't fight the system** - JSON + heredocs don't mix; use tools that work with JSON

4. **Measure trade-offs** - Eliminated one error but introduced another (net negative)

5. **EAFP works for AI systems** - Accepting and ignoring is more robust than restricting

---

**Status:** Analysis complete, ready to implement **kwargs approach
**Next step:** Revert bash migration, add **kwargs, re-run evaluation
