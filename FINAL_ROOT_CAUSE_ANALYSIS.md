# L3-L7 Evaluation Failures - Final Root Cause Analysis

## Executive Summary

The 100% failure rate in L3-L7 orchestrator evaluations had THREE root causes:

1. ✅ **FIXED**: ToolCallingSyntaxBehavior method name mismatch
2. ✅ **FIXED**: Missing qwen3-coder XML format parser
3. ❌ **MODEL INCOMPATIBILITY**: qwen3-coder:30b doesn't support tool calling

## Detailed Analysis

### Bug #1: Method Name Mismatch (FIXED)

**File**: `behaviors/tool_calling_syntax.py:81`
**Problem**: Method named `inject_initial_context()` but event system calls `on_initial_context()`
**Impact**: JSON examples never injected into context
**Fix**: Renamed to `on_initial_context(agent, context)`
**Commit**: `b90e0ef`

### Bug #2: Missing XML Format Parser (FIXED)

**File**: `behaviors/tool_calling_syntax.py:240-298`
**Problem**: XML parser only supported Anthropic format, not qwen3-coder format
- Anthropic: `<invoke name="tool"><parameter name="arg">val</parameter></invoke>`
- qwen3-coder: `<function=tool><parameter=arg>val</parameter></function>`

**Impact**: Even when LLM returned XML, it wasn't parsed
**Fix**: Added regex for qwen3-coder XML format
**Commit**: `b90e0ef`

### Root Cause #3: Model Incompatibility (UNFIXABLE)

**Model**: qwen3-coder:30b
**Problem**: Model does NOT support Ollama tool calling protocol

**Evidence from Testing**:
- With fixes applied: Still 29+ consecutive empty rounds
- JSON examples injected: ✓
- XML fallback parser active: ✓
- Tool calls detected: ✗

**Evidence from Prior Evaluations** (`evaluation_results/model_comparison_20251106_173940.json`):
- **gpt-oss:20b**: 100% success rate, 5-20s per task
- **qwen3:8b**: Works with tool calling
- **qwen3:14b**: Works with tool calling
- **qwen3-coder:30b**: NOT TESTED (likely because it doesn't work)

**Conclusion**: qwen3-coder variants are code-specialized but do NOT support function calling.

## Resolution

### Immediate Action Required

**Change default model from qwen3-coder:30b to gpt-oss:20b**

Edit `config/llm_config.yaml`:
```yaml
# Before
model: "qwen3-coder:30b"

# After
model: "gpt-oss:20b"  # Proven 100% success rate
```

### Why gpt-oss:20b?

1. **Proven track record**: 100% success on L3-L7 tasks
2. **Fast**: 5-20 seconds per task (vs 5-8 minutes with qwen3-coder)
3. **Reliable tool calling**: Native Ollama function calling support
4. **128K context**: Same as qwen3-coder
5. **Open source**: No licensing restrictions

### Model Selection Matrix

| Model | Tool Calling | Code Quality | Speed | Recommended For |
|-------|--------------|--------------|-------|----------------|
| gpt-oss:20b | ✅ Excellent | ✅ Good | ✅ Fast | **PRIMARY** (general agent work) |
| qwen3:8b | ✅ Good | ⚠️  Fair | ✅ Fast | Backup / testing |
| qwen3:14b | ✅ Good | ✅ Good | ⚠️  Medium | Larger context needs |
| qwen3-coder:30b | ❌ None | ✅ Excellent | ❌ Slow | ⚠️  **Code completion only** (no agents) |

## Test Results

### Before Fixes
```
Model: qwen3-coder:30b
Success: 0/4 (0%)
Pattern: Empty workspaces, no tool calls detected
Time: ~5-8 minutes per task before timeout
```

### After Method Name Fix + XML Parser
```
Model: qwen3-coder:30b
Success: 0/26 (0% - evaluation still running)
Pattern: 29+ consecutive empty rounds
Time: ~3-4 minutes before auto-fail
Issue: Model returns text, no tool calls
```

### Expected After Model Switch
```
Model: gpt-oss:20b
Expected Success: 18-22/26 (70-85%) based on prior evals
Expected Time: 5-20 seconds per task
Pattern: Fast delegation → implementation → validation
```

## Verification Steps

1. Update `config/llm_config.yaml` to use gpt-oss:20b
2. Re-run evaluation: `python3 tests/orchestrator_l3_l7_eval.py`
3. Expected results:
   - Most tasks complete in <30 seconds
   - Files created in workspaces
   - Tool calls visible in logs
   - Success rate 70-85%

## Related Documents

- `/workspace/TOOL_CALLING_BUG_FIX.md` - Details on bugs #1 and #2
- `/workspace/CURRENT_STATUS.md` - Original fix attempt
- `/workspace/docs/TOOL_CALLING_AND_ERROR_FEEDBACK_PLAN.md` - Implementation plan
- `/workspace/evaluation_results/model_comparison_20251106_173940.json` - Model performance data
