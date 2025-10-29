# Implemented Parameters: LLM Hallucinations as Feature Requests

**Date:** 2025-10-29
**Approach:** Treat parameter hallucinations as feature requests and implement them

## Analysis of Hallucinated Parameters

From evaluation logs, the LLM attempted to use these parameters:

| Parameter | Occurrences | Function | Status |
|-----------|-------------|----------|--------|
| `line_end` | 12 | write_file | ✅ **IMPLEMENTED** |
| `overwrite` | 3 | write_file | ✅ **IMPLEMENTED** |
| `timeout` | 1 | write_file | ❌ Ignored (doesn't make sense for sync I/O) |
| `content` | 1 | read_file | ❌ Ignored (parameter already exists) |
| `status` | 2 | Various | ❌ Ignored (not file-related) |

## Implemented Parameters

### write_file: 5 new parameters + **kwargs

```python
def write_file(
    path: str,              # Required
    content: str,           # Required
    append: bool = False,   # NEW: Append instead of overwrite
    encoding: str = "utf-8", # NEW: Custom text encoding
    line_end: str | None = None,  # NEW: Line ending control (most requested!)
    overwrite: bool = True, # NEW: Fail if file exists
    **kwargs               # NEW: Gracefully ignore other parameters
) -> str:
```

#### 1. line_end (most requested - 12 occurrences)

**Use case:** Cross-platform line ending control

**Examples:**
```python
# Unix line endings (default Python behavior)
write_file("script.sh", "#!/bin/bash\necho hello", line_end="\n")

# Windows line endings (.bat files, Windows text files)
write_file("script.bat", "@echo off\necho hello", line_end="\r\n")

# Normalize mixed line endings to Unix
write_file("normalized.txt", "line1\r\nline2\rline3\n", line_end="\n")
# Result: All line endings become \n
```

**Implementation:**
1. Normalize all line endings to `\n`
2. Replace with desired line ending
3. Use `newline=''` to prevent Python from re-translating

**Verified:** Binary file inspection confirms correct line endings on disk

#### 2. overwrite (3 occurrences)

**Use case:** Prevent accidental file overwrites

**Examples:**
```python
# Safe write - fail if file exists
write_file("important.json", "{...}", overwrite=False)
# Returns error if file exists

# Force overwrite (default behavior)
write_file("temp.txt", "data", overwrite=True)
```

**Implementation:** Check `Path.exists()` before writing

#### 3. append (new, but related to common use case)

**Use case:** Add to file without reading/rewriting

**Examples:**
```python
# Initialize log file
write_file("app.log", "Log started\n")

# Append entries
write_file("app.log", "Entry 1\n", append=True)
write_file("app.log", "Entry 2\n", append=True)
```

**Implementation:** Use `"a"` mode instead of `"w"` mode

#### 4. encoding (new, but reasonable)

**Use case:** Handle non-UTF-8 files

**Examples:**
```python
# Write Latin-1 encoded file
write_file("legacy.txt", "Café résumé", encoding="latin-1")

# Write UTF-16 file
write_file("unicode.txt", "Hello 世界", encoding="utf-16")
```

**Implementation:** Pass encoding to `open()`

#### 5. **kwargs (catch-all)

**Use case:** Gracefully handle future parameter invention

**Examples:**
```python
# LLM invents "timeout" parameter
write_file("file.py", "code", timeout=60)
# Prints: [tools] write_file ignoring unsupported parameters: timeout
# Still writes file successfully

# LLM invents "mode" parameter
write_file("file.py", "code", mode="w")
# Prints warning, ignores parameter, writes file
```

**Implementation:** Accept **kwargs, log ignored parameters, continue execution

### read_file: 2 new parameters + **kwargs

```python
def read_file(
    path: str,                    # Required
    encoding: str = "utf-8",      # NEW: Custom text encoding
    max_size: int = 1_000_000,    # NEW: Configurable read limit
    **kwargs                      # NEW: Gracefully ignore other parameters
) -> str:
```

#### 1. encoding

**Use case:** Read non-UTF-8 files

**Examples:**
```python
# Read Latin-1 encoded file
content = read_file("legacy.txt", encoding="latin-1")

# Read UTF-16 file
content = read_file("unicode.txt", encoding="utf-16")
```

#### 2. max_size

**Use case:** Adjust read limit for different use cases

**Examples:**
```python
# Read very large file (10MB limit)
content = read_file("huge.log", max_size=10_000_000)

# Read small snippet (100KB limit)
content = read_file("partial.txt", max_size=100_000)
```

#### 3. **kwargs

Same as write_file - gracefully ignore unexpected parameters.

## Parameters NOT Implemented (and why)

### timeout (write_file)

**Why not:** File writes are synchronous and complete in milliseconds. A timeout doesn't make sense for standard file I/O.

**Alternative:** For slow writes (network drives, etc.), the OS handles timeouts. If truly needed, use `run_bash` with timeout parameter.

### content (read_file)

**Why not:** This is already a required parameter name. The hallucination might have been for a filter/search parameter, but that's too ambiguous.

**Alternative:** Use `run_bash("grep 'pattern' file.txt")` for content filtering.

### status (various)

**Why not:** Not file-operation related. Likely confused with subtask status.

**Alternative:** Use `mark_subtask_complete(success=True)` for status reporting.

## Benefits of Implementation Approach

### 1. No More Crashes

**Before:**
```python
write_file("file.py", "code", line_end="\n")
# TypeError: write_file() got an unexpected keyword argument 'line_end'
# Agent crashes, task fails
```

**After:**
```python
write_file("file.py", "code", line_end="\n")
# Uses parameter, writes file with Unix line endings
# Agent continues successfully
```

### 2. Better User Experience

**Before:** LLM had to use bash heredocs:
```python
run_bash("cat > file.py <<'EOF'\ncode\nEOF")
# 150+ characters, complex syntax, JSON parsing issues
```

**After:** LLM uses simple function call:
```python
write_file("file.py", "code")
# 35 characters, simple syntax, no JSON issues
```

### 3. More Powerful Tools

The implemented parameters are actually useful:
- **line_end**: Essential for cross-platform files
- **append**: Common use case for logs
- **overwrite**: Safety feature
- **encoding**: Necessary for international/legacy files
- **max_size**: Performance control

### 4. Future-Proof

New hallucinations are handled gracefully:
- Logged for analysis
- Don't crash the agent
- Can be promoted to real parameters if frequently used

## Testing

### Smoke Test ✅

```python
# Test basic functionality
agent = TaskExecutorAgent(
    goal="Create hello.py with print('Hello, World!')",
    max_rounds=15
)
# Result: ✓ PASSED (6 rounds)
```

### Parameter Tests ✅

```python
# Unix line endings
write_file('test.txt', 'line1\nline2', line_end='\n')
# Binary: b'line1\nline2' ✓

# Windows line endings
write_file('test.txt', 'line1\nline2', line_end='\r\n')
# Binary: b'line1\r\nline2' ✓

# Mixed input, normalized to Unix
write_file('test.txt', 'line1\r\nline2\rline3\n', line_end='\n')
# Binary: b'line1\nline2\nline3\n' ✓

# Append mode
write_file('log.txt', 'Entry 1\n')
write_file('log.txt', 'Entry 2\n', append=True)
# Content: "Entry 1\nEntry 2\n" ✓

# Overwrite protection
write_file('important.txt', 'data')
write_file('important.txt', 'new', overwrite=False)
# Returns error ✓
```

### L5-L7 Evaluation ⏳

Early results (3/3 L5 blog_system):
- ✅ No parameter invention crashes
- ✅ No JSON parsing errors
- ✅ Clean execution with proper parameters

Full results pending completion.

## Comparison: Before vs After

| Metric | Before (7.4%) | Bash Migration (59% JSON errors) | After (Expected) |
|--------|---------------|-----------------------------------|------------------|
| **Errors** ||||
| Parameter invention | 67% | 0% | 0% |
| JSON parsing | 0% | 59% | 0% |
| **Ergonomics** ||||
| Chars per write | ~50 | ~150 | ~50 |
| Syntax complexity | Low | High | Low |
| Human readable | Yes | No | Yes |
| **Capabilities** ||||
| Line ending control | No | No | **Yes** ✅ |
| Append mode | No | Yes | **Yes** ✅ |
| Overwrite protection | No | No | **Yes** ✅ |
| Custom encoding | No | Yes | **Yes** ✅ |
| **Maintainability** ||||
| Code complexity | Medium | High | Low |
| Edge cases | Many | Very many | Few |
| Future-proof | No | No | **Yes** ✅ |

## Key Insights

### 1. LLM Hallucinations Are Signal

When the LLM consistently invents the same parameter (like `line_end` 12 times), it's telling us:
- This feature makes sense
- This parameter is logical
- Users will expect this capability

**Treat hallucinations as feature requests, not bugs.**

### 2. Implement What Makes Sense

Not all hallucinated parameters should be implemented:
- ✅ `line_end`: Useful, common need, well-defined
- ✅ `overwrite`: Safety feature, clear semantics
- ❌ `timeout`: Doesn't make sense for sync file I/O
- ❌ `content` (read_file): Ambiguous, unclear purpose

### 3. **kwargs Is Essential

Even with implemented parameters, LLM may invent new ones:
- **kwargs prevents crashes
- Logging helps identify new patterns
- Easy to promote frequent inventions to real parameters

### 4. Simple Beats Clever

We tried "clever" solutions (bash heredocs) that failed. The simple solution (just implement the parameters) works best:
- No JSON incompatibilities
- No complex syntax to learn
- No edge cases to handle
- Just straightforward Python

### 5. User Feedback Is Valuable

User suggestion: "implement common **kwargs the agent was attempting to use"

This led to:
- Analyzing actual hallucination patterns
- Implementing line_end (12 occurrences - would have missed this!)
- Creating more useful tools overall

## Recommendations

### For This Project

1. ✅ Keep **kwargs - essential safety net
2. ✅ Monitor warnings for new patterns
3. ⏭️ If a parameter appears 5+ times, consider implementing it
4. ⏭️ Document all implemented parameters clearly

### For Future AI Tool Design

1. **Start with **kwargs** - Make tools tolerant by default
2. **Log parameter usage** - Understand what AI wants
3. **Implement top N** - Add most-requested parameters
4. **Keep simple** - Don't over-engineer restrictions
5. **Test with AI** - Validate with actual LLM usage

## Next Steps

1. ⏳ Wait for L5-L7 evaluation to complete
2. ⏳ Analyze usage patterns of implemented parameters
3. ⏳ Check for new parameter inventions in **kwargs warnings
4. ⏳ Document success rate improvement
5. ⏳ Update CLAUDE.md with new capabilities

---

**Status:** Implementation complete, testing in progress
**Parameters implemented:** 7 new parameters + 2 **kwargs safety nets
**Lines changed:** ~150 (mostly documentation and implementation)
**Complexity:** Low (straightforward additions)
**Expected impact:** Eliminate ~80% of failures (parameter + JSON errors)
