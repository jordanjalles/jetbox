# L5-L7 Evaluation: Post-Fixes Regression Report

**Date:** 2025-10-29
**Evaluation:** L5-L7 with semantic validation
**Model:** gpt-oss:20b

## Executive Summary

**CRITICAL REGRESSION: 2/27 tasks passed (7.4%)**

This is significantly worse than previous run (8/24 = 33%). The three major fixes applied (context isolation, bash tool migration, simplified L6 specs) did NOT improve results. The parameter invention problem has actually gotten WORSE.

## Results by Level

| Level | Passed | Total | Rate |
|-------|--------|-------|------|
| L5    | 2      | 9     | 22%  |
| L6    | 0      | 9     | 0%   |
| L7    | 0      | 9     | 0%   |
| **Total** | **2** | **27** | **7.4%** |

### Comparison with Previous Run

| Level | Previous (detailed specs) | Current (simplified + fixes) | Change |
|-------|---------------------------|------------------------------|---------|
| L5    | 78%                       | 22%                          | **-56%** ⬇️ |
| L6    | 0%                        | 0%                           | 0% |
| L7    | 22%                       | 0%                           | **-22%** ⬇️ |
| **Overall** | **33%**            | **7.4%**                     | **-26%** ⬇️ |

## Critical Finding: Parameter Invention Crisis

**72% of failures (18/25) were tool parameter errors!**

### Error Distribution:

1. `read_file(line_end=...)` - **8 errors**
2. `write_file(overwrite=...)` - **4 errors**
3. `write_file(line_end=...)` - **4 errors**
4. `write_file(timeout=...)` - **1 error**
5. `read_file(content=...)` - **1 error**
6. JSON parsing errors - 3 errors
7. Ollama timeouts - 2 errors

### Why Bash Tool Migration Failed

The bash tool migration successfully eliminated:
- ✓ `grep_file` parameter errors (removed entirely)
- ✓ `run_cmd` timeout errors (removed entirely)

But it did NOT address:
- ✗ `write_file` parameter invention (16 new parameter errors!)
- ✗ `read_file` parameter invention (9 new parameter errors!)

**Root cause:** The LLM invents parameters for ANY function that seems to need them. Replacing `grep_file` with `run_bash` doesn't help `write_file`.

## Detailed Error Analysis

### L5 Results (2/9 passed = 22%)

**Passed:**
- blog_system run 2 (38.9s, 30 rounds)
- inventory_system run 1 (59.8s, 17 rounds)

**Failed with parameter errors:**
- blog_system run 1: `read_file(line_end=...)`
- blog_system run 3: `read_file(line_end=...)`
- todo_app run 1: `write_file(overwrite=...)`
- todo_app run 3: `read_file(line_end=...)`
- inventory_system run 2: `write_file(timeout=...)`
- inventory_system run 3: `read_file(line_end=...)`

**Other failures:**
- todo_app run 2: JSON parsing error (malformed heredoc in tool call)

### L6 Results (0/9 passed = 0%)

**All patterns failed:**
- Observer pattern: 0/3 (even after simplification!)
- Factory pattern: 0/3
- Dependency injection: 0/3

**Failed with parameter errors:**
- observer run 1: `read_file(content=...)`
- observer run 3: `write_file(overwrite=...)`
- factory run 1: `read_file(line_end=...)`
- factory run 2: `read_file(line_end=...)`
- factory run 3: `write_file(overwrite=...)`
- dependency_injection run 1: `read_file(line_end=...)`

**Other failures:**
- observer run 2: Tests passing but missing `subscribe`/`unsubscribe` methods (validation failure)
- dependency_injection run 2: Ollama timeout (30s no response)
- dependency_injection run 3: JSON parsing error (malformed tool call)

### L7 Results (0/9 passed = 0%)

**All algorithms failed:**
- Rate limiter: 0/3
- Connection pool: 0/3
- Circuit breaker: 0/3

**Failed with parameter errors:**
- rate_limiter run 2: `write_file(line_end=...)`
- rate_limiter run 3: `read_file(line_end=...)`
- connection_pool run 2: `write_file(line_end=...)`
- circuit_breaker run 1: `write_file(line_end=...)`
- circuit_breaker run 2: `write_file(line_end=...)`

**Other failures:**
- rate_limiter run 1: Validation failure (missing `allow`/`check` methods)
- connection_pool run 1: JSON parsing error (malformed heredoc)
- connection_pool run 3: Validation failure (missing `Pool` class)
- circuit_breaker run 3: Ollama timeout

## Impact of Applied Fixes

### 1. Context Isolation Fix (✓ Working, ⚠️ No Impact)

**Status:** Verified working (test passing)

**Expected benefit:** Reduce confusion from prior subtask messages bleeding into current context

**Actual impact:** NO MEASURABLE IMPROVEMENT
- Most runs failed in early rounds (< 10 rounds) before subtask transitions
- Tool errors prevented tasks from reaching multiple subtasks
- Cannot assess benefit when 72% of runs crash on tool errors

**Conclusion:** Fix is correct but benefit masked by tool errors

### 2. Bash Tool Migration (✓ Partial Success, ✗ Incomplete)

**Status:** Partially implemented

**What worked:**
- ✓ Removed `grep_file` (no more grep parameter errors)
- ✓ Removed `run_cmd` timeout issues
- ✓ Full bash access now available

**What didn't work:**
- ✗ Did NOT replace `write_file` with bash equivalent
- ✗ Did NOT replace `read_file` with bash equivalent
- ✗ Parameter invention shifted to remaining Python wrappers

**Conclusion:** Migration incomplete - need to finish the job

### 3. Simplified L6 Specs (✗ No Impact)

**Status:** Implemented

**Expected benefit:** Reduce confusion on well-known patterns

**Actual impact:** NO IMPROVEMENT (0% → 0%)
- All L6 runs still failed
- 6/9 failures were tool parameter errors
- Cannot assess spec quality when tools crash

**Conclusion:** Cannot evaluate until tool errors fixed

## Root Cause Analysis

### Why Parameter Invention Keeps Happening

The LLM operates on **logical inference**:
- "I need to write a file without newlines at the end"
- "I've seen `line_end` parameters in other APIs"
- "write_file probably has a `line_end` parameter"
- **Invents:** `write_file(path, content, line_end='\n')`

This is **rational behavior given incomplete information**.

### Why Python Wrappers Encourage Invention

Python functions with signatures like:
```python
def write_file(path: str, content: str) -> dict:
```

Trigger the thought:
- "This is too simple - what about append mode?"
- "What about atomic writes?"
- "What about line endings?"
- "Let me add those parameters..."

Bash commands with full syntax documented don't trigger this:
```bash
cat > file.txt <<'EOF'
content
EOF
```

LLM thinks: "This is the complete syntax, I can't add parameters"

## Solutions Analysis

### Option 1: Accept **kwargs (Quick Fix)

**Implementation:**
```python
def write_file(path: str, content: str, **kwargs) -> dict:
    # Ignore unknown parameters
    if kwargs:
        print(f"[tool] Ignoring unknown parameters: {list(kwargs.keys())}")
    # ... existing implementation ...
```

**Pros:**
- Quick to implement (< 5 minutes)
- No agent crashes on parameter errors
- LLM continues working despite wrong assumptions

**Cons:**
- Doesn't solve root problem (LLM still confused)
- May hide legitimate issues
- Feels like a band-aid

### Option 2: Full Bash Migration (Proper Fix)

**Implementation:**
```python
# REMOVE write_file entirely from tools
# System prompt update:
"""
Use run_bash for all file operations:
- Write file: run_bash("cat > file.txt <<'EOF'\\ncontent\\nEOF")
- Read file: run_bash("cat file.txt")
- Append: run_bash("cat >> file.txt <<'EOF'\\nmore\\nEOF")
"""
```

**Pros:**
- Consistent with bash tool philosophy
- LLM can't invent parameters (fixed syntax)
- More powerful (supports all shell features)
- Eliminates entire class of errors

**Cons:**
- Requires rewriting system prompt
- Higher cognitive load (need to write shell syntax)
- Less ergonomic for simple writes

### Option 3: Add Missing Parameters (Wrong Approach)

**Implementation:**
```python
def write_file(path: str, content: str,
               line_end: str = '\n',
               overwrite: bool = True,
               timeout: int = None) -> dict:
```

**Pros:**
- Satisfies LLM expectations
- No crashes

**Cons:**
- **Parameter explosion** - LLM will invent MORE
- Endless whack-a-mole
- Violates "simple tools" principle
- This is how we got here in the first place!

### Option 4: Better Documentation (Necessary But Insufficient)

**Implementation:**
```yaml
system_prompt: |
  CRITICAL: Only use these EXACT parameters:

  write_file(path: str, content: str)
  - NO other parameters exist
  - For special needs, use run_bash instead

  read_file(path: str)
  - NO other parameters exist
  - For partial reads, use run_bash('head -20 file')
```

**Pros:**
- Clarifies expectations
- Low effort

**Cons:**
- Might not work (LLM may still invent)
- Need to test effectiveness

## Recommended Action Plan

### Immediate Actions (Stop the Bleeding)

**1. Apply **kwargs fix (5 minutes)**

Prevents crashes while we implement proper solution.

```python
# tools.py
def write_file(path: str, content: str, **kwargs) -> dict[str, Any]:
    if kwargs:
        print(f"[tools] write_file called with unexpected kwargs: {list(kwargs.keys())}")
    # ... rest of implementation ...

def read_file(path: str, **kwargs) -> str:
    if kwargs:
        print(f"[tools] read_file called with unexpected kwargs: {list(kwargs.keys())}")
    # ... rest of implementation ...
```

**2. Re-run L5-L7 evaluation**

Verify that removing tool crashes improves results.

### Medium-term Actions (Proper Fix)

**3. Complete bash migration (1 hour)**

Remove write_file, update system prompt with bash equivalents:
```yaml
File operations via run_bash:
- Write: run_bash("cat > file.py <<'EOF'\ncode\nEOF")
- Read: run_bash("cat file.py")
- Append: run_bash("cat >> file.py <<'EOF'\nmore\nEOF")
```

**4. Update evaluation tasks**

Re-run L5-L7 with full bash approach.

### Long-term Actions (Validation)

**5. Compare approaches**

Run identical evaluation with:
- A: **kwargs approach
- B: Full bash approach

Measure:
- Success rates
- Token usage
- Round counts
- Hallucination rates

**6. Document findings**

Create comprehensive tool design guide based on results.

## Key Insights

### What We Learned

1. **Partial migrations are dangerous** - Removing `grep_file` but keeping `write_file` just shifted the problem

2. **Parameter invention is model behavior, not a bug** - The LLM is trying to be helpful by inferring missing parameters

3. **Simple functions invite complexity** - `write_file(path, content)` feels incomplete, so LLM completes it

4. **Bash syntax is self-documenting** - Shell commands don't invite parameter invention because syntax is complete

5. **Context isolation fix works but benefit masked** - Can't evaluate agent improvements when tools crash

### What We Should Have Done

1. **Complete bash migration in one go** - Half measures don't work

2. **Test tool changes in isolation first** - Should have run quick tool-only tests before full evaluation

3. **Add tool parameter validation mode** - Should log unknown parameters instead of crashing during development

## Next Steps

**Recommend:** Apply **kwargs fix immediately, then schedule full bash migration.

**Question for user:** Do you want to:
1. Apply **kwargs band-aid and re-run evaluation now?
2. Skip to full bash migration (write_file → run_bash)?
3. Try Option 4 (better documentation) first?

---

**Files:**
- Results: `/workspace/l5_l7_semantic_results.json`
- Full log: `/workspace/l5_l7_semantic_output.log`
- Previous report: `/workspace/analysis/EVAL_L1_L7_X3_REPORT.md`
