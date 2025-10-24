# Stress Test Cleanup Fix

**Date:** 2025-10-23
**Issue:** `clean_workspace()` was deleting critical files from root directory

---

## Problem

The stress test script's `clean_workspace()` function was too aggressive:

```python
# OLD CODE (DANGEROUS):
for item in Path(".").iterdir():
    if item.is_file() and item.suffix in ['.py', '.txt', '.json', '.log']:
        if item.name not in preserve_files:
            item.unlink()  # Deletes files in ROOT!
```

**What went wrong:**
1. Function iterates over ALL files in root directory
2. Deletes any `.py` file not in a hardcoded whitelist
3. `agent_config.py` was initially missing from whitelist
4. Even after adding it, the preserve list was fragile
5. Any new agent file would get deleted unless explicitly added

**Impact:**
- `agent_config.py` kept getting deleted between test runs
- Tests failed with `ModuleNotFoundError: No module named 'agent_config'`
- Had to recreate the file multiple times
- Made iterative development impossible

---

## Solution

**Removed all root-level file deletion:**

```python
# NEW CODE (SAFE):
# REMOVED: DO NOT delete files in root - too dangerous!
# The agent creates files in .agent_workspace which gets cleaned above
# Test artifacts stay in workspace, not root
```

**What it now cleans (ONLY):**
1. `.agent_workspace/` - Agent's isolated workspace directory
2. `.agent_context/` - Agent's state/history directory
3. `agent_v2.log`, `agent_ledger.log`, `agent.log` - Specific log files
4. `__pycache__/` directories - Python bytecode cache

**What it does NOT touch:**
- ✅ Source files in root (`.py`, `.yaml`, etc.)
- ✅ Documentation files
- ✅ Configuration files
- ✅ Test scripts themselves
- ✅ Any other root-level content

---

## Why This is Better

### Safety
- No risk of deleting important files
- No need to maintain fragile whitelist
- New files don't need to be added to preserve list

### Correctness
- Agent uses `.agent_workspace/` for isolated work
- That directory gets cleaned - exactly what we want
- Source code stays untouched - exactly what we want

### Maintainability
- One less thing to worry about
- Future agent files work automatically
- No "preserve_files" list to update

---

## File Modified

**File:** `run_stress_tests.py`
**Lines:** 278-280 (replaced 17 dangerous lines with 3-line comment)
**Change:**
```diff
- # Remove test artifact files in root
- for item in Path(".").iterdir():
-     if item.name.startswith('.') and item.name != '.git':
-         continue
-     if item.is_file() and item.suffix in ['.py', '.txt', '.json', '.log']:
-         if item.name not in preserve_files:
-             try:
-                 item.unlink()
-             except Exception:
-                 pass
-     elif item.is_dir() and item.name not in {...}:
-         if not item.name.startswith('.'):
-             try:
-                 shutil.rmtree(item)
-             except Exception:
-                 pass

+ # REMOVED: DO NOT delete files in root - too dangerous!
+ # The agent creates files in .agent_workspace which gets cleaned above
+ # Test artifacts stay in workspace, not root
```

---

## Lessons Learned

1. **Never clean root programmatically** - Too risky, too fragile
2. **Use isolated directories for tests** - Clean those, not root
3. **Whitelists are dangerous** - Easy to forget entries
4. **Explicit is better** - Only delete what you create

---

## Testing

After fix:
```bash
$ ls agent_config.py
-rw-r--r-- 1 root root 2686 Oct 23 01:16 agent_config.py

$ python -c "import agent_config; print('✓ Import works')"
✓ Import works

$ python agent.py "test task"
[works correctly - no import errors]
```

The agent now runs reliably without file deletion issues.
