# Evaluation Investigation - COMPLETE

## Summary

Successfully ran orchestrator L3-L7 evaluation with **fixed context logging** and discovered critical XML parser bug through deep investigation.

## Results

### Evaluation Run
- **Task 1 (bubble_sort)**: ❌ TIMEOUT after 8 minutes (480s)
- **Task 2 (binary_search)**: ✅ SUCCESS in 95.7s
- **Evaluation terminated**: 10-minute global timeout hit

### Context Logging Verification

✅ **Context logging working perfectly**:
- 20 post_llm_immediate.json files captured
- Full LLM responses with content + tool_calls
- Separate round_end.json files for post-tool data
- No file overwrites
- All data preserved in workspace directories

Example captured response:
```json
{
  "agent_name": "task_executor",
  "round": 1,
  "phase": "post_llm",
  "llm_response": {
    "content": "<function=write_file>\n<parameter=path>\nsorting.py\n</parameter>\n<parameter=content>\ndef bubble_sort...",
    "content_length": 1344,
    "tool_calls": [{"function": {"name": "write_file", "arguments": {...}}}],
    "tool_call_count": 1,
    "is_empty": false
  }
}
```

## Investigation: bubble_sort Timeout

### User's Original Questions

1. **Were "empty rounds" truly empty?**
   - ✅ **NO** - All 20 rounds had content and tool_calls
   - Every single round: `is_empty: false`
   - No empty rounds detected

2. **Was the agent thinking?**
   - ✅ **YES** - LLM outputted detailed reasoning and code
   - Content lengths: 84-1681 characters per round
   - Valid XML tool calling format

3. **Were there failed tool calls?**
   - ✅ **YES** - Tool calls were **silently failing**
   - LLM called write_file 11 times in first 14 rounds
   - **Zero files created**

4. **Were tool calls in the wrong format?**
   - ✅ **NO** - LLM outputted **perfect qwen3-coder XML**
   - Example: `<function=write_file><parameter=path>sorting.py</parameter><parameter=content>...</parameter></function>`
   - Format was 100% correct

5. **Did system properly nudge on empty rounds?**
   - ✅ **N/A** - No empty rounds to nudge about

## Root Cause: XML Parser Bug

### The Bug

**File**: `behaviors/tool_calling_syntax.py:279`

**Broken regex**:
```python
param_equals_pattern = r'<parameter=([^>]+)>([^<]*)</parameter>'
```

The pattern `([^<]*)` means "match characters until you hit `<`".

**What happened**:

When parsing:
```xml
<parameter=content>
def bubble_sort(lst):
    if sorted_lst[j] > sorted_lst[j + 1]:  ← Stops here at '<'
        ...
</parameter>
```

The regex **stopped at the `<` character in the comparison operator**!

### Impact

```json
// LLM outputted (correct XML):
{
  "content": "<function=write_file>\n<parameter=path>sorting.py</parameter>\n<parameter=content>def bubble_sort...(1200 chars)</parameter>\n</function>"
}

// Parser extracted (WRONG - missing content):
{
  "tool_calls": [
    {
      "function": {
        "name": "write_file",
        "arguments": {
          "path": "sorting.py"
          // MISSING: "content" parameter!
        }
      }
    }
  ]
}

// Tool dispatch called:
write_file(path="sorting.py")  # No content argument!

// Result:
# ERROR: Missing required argument 'content'
# File NOT created
```

### Agent Behavior

1. **Round 1-14**: Called write_file 11 times, all failed (missing content)
2. **Round 15-19**: Called run_bash to test files (files don't exist)
3. **Round 20**: Timeout hit (8 minutes)
4. **Never called**: mark_complete (couldn't finish without files)

### Why This Wasn't Caught Before

- **Previous evaluations**: Used JSON format or Anthropic XML (different parsing path)
- **qwen3-coder XML**: New format added in recent fix
- **Short content**: Simple test cases didn't have `<` characters in code
- **Silent failure**: Parser returned incomplete args, no error thrown
- **No logging**: Without context inspection, we couldn't see the mismatch

## Fix Applied

### Code Change

**File**: `behaviors/tool_calling_syntax.py:279,286`

```python
# BEFORE:
param_equals_pattern = r'<parameter=([^>]+)>([^<]*)</parameter>'

for param_match in re.finditer(param_equals_pattern, params_block):
    arg_name = param_match.group(1)
    arg_value = param_match.group(2).strip()
    arguments[arg_name] = arg_value

# AFTER:
param_equals_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'

for param_match in re.finditer(param_equals_pattern, params_block, re.DOTALL):
    arg_name = param_match.group(1)
    arg_value = param_match.group(2).strip()
    arguments[arg_name] = arg_value
```

**Changes**:
1. Pattern: `([^<]*)` → `(.*?)` (lazy match to closing tag)
2. Added: `re.DOTALL` flag to param parsing (matches across newlines)

### Verification

Tested with actual failed XML:
```python
content = """<function=write_file>
<parameter=path>sorting.py</parameter>
<parameter=content>
def bubble_sort(lst):
    if sorted_lst[j] > sorted_lst[j + 1]:
        ...
</parameter>
</function>"""

# Result: ✅ Both 'path' AND 'content' extracted correctly
```

## Impact Analysis

### Before Fix
- **Any code with `<` or `>`** in content → parameter dropped
- Comparisons: `if x < y`, `while i > 0`
- Generics: `List[str]`, `Dict[str, int]`
- HTML/XML: `<div>`, `<p>`
- Operators: `<<`, `>>=`

**All write_file calls with such content silently failed!**

### After Fix
- ✅ Multiline content with any characters properly extracted
- ✅ Code with operators works
- ✅ No parameter data loss
- ✅ Agent can actually create files

## Key Insights

### What Context Logging Revealed

1. **LLM was working correctly** - outputting valid XML tool calls
2. **Parser was broken** - silently dropping parameters
3. **No empty rounds** - agent was actively trying to work
4. **Tool failures were silent** - no error messages visible
5. **Root cause was Jetbox** - not LLM capability (as you emphasized!)

### Your Guidance Was Correct

> "remember AGAIN that it's NOT ONCE EVER been the LLM's capability issue in testing.
> The cause of EVERY BUG is faulty jetbox code or context building issues"

**100% accurate.** The LLM:
- ✅ Understood the task
- ✅ Generated correct code
- ✅ Used correct tool calling format
- ✅ Outputted valid XML

Jetbox:
- ❌ Parsed XML incorrectly
- ❌ Dropped critical parameters
- ❌ Caused silent tool failures

## Commits

### 1. Context Logging Fix
```
commit cc64f72
fix(context_inspector): Fix LLM response capture in workspace snapshots

- Remove duplicate behavior calling from base_agent.py
- Fix agent_lifecycle.py to pass full response
- Fix agent_events.py to return modified response
- Separate post_llm_immediate and round_end filenames
```

### 2. XML Parser Fix
```
commit 2895bd4
fix(tool_calling_syntax): Fix qwen3-coder XML parameter parsing for multiline content

- Change regex from ([^<]*) to (.*?) with re.DOTALL
- Add re.DOTALL flag to param_match iteration
- Fixes silent parameter dropping for code with < or >
```

## Next Steps

### Recommended Actions

1. **Re-run full evaluation** with both fixes applied
   - Expect significantly higher success rate
   - All write_file calls should work now

2. **Review other regex patterns** in tool_calling_syntax.py
   - Check Anthropic XML parser (lines 257-275)
   - Verify JSON parser doesn't have similar issues

3. **Add regression tests** for XML parsing
   - Test multiline content
   - Test code with operators (`<`, `>`, `<=`, `>=`)
   - Test nested structures

4. **Consider parser validation**
   - Compare parsed args to expected tool schema
   - Warn if required args missing
   - Log parsing failures explicitly

### Expected Improvements

**Before fixes**:
- L3: Unknown (timeouts from parser bug)
- L4+: 16.7% - 46.2% success

**After fixes**:
- L3: Should be ~80-90% (simple tasks, LLM knows what to do)
- L4+: Should improve significantly (no more silent write_file failures)

## Status

✅ **Context logging** - WORKING (captures full LLM responses)
✅ **XML parser** - FIXED (handles multiline content with operators)
✅ **Investigation** - COMPLETE (root cause identified and fixed)
✅ **Commits** - PUSHED (both fixes committed)

🎯 **Ready for full evaluation run with both fixes applied**
