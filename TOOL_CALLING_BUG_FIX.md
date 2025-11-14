# Tool Calling Bug Fix - Root Cause Analysis

## Summary

The ToolCallingSyntaxBehavior was completely non-functional due to two critical bugs that prevented it from injecting JSON examples or parsing XML tool calls from qwen3-coder:30b.

## Bug #1: Wrong Event Hook Name

**Location**: `behaviors/tool_calling_syntax.py:81`

**Problem**: Method was named `inject_initial_context()` but the event system calls `on_initial_context()`.

**Impact**:
- JSON format examples were NEVER injected into agent context
- Agents received no guidance on how to format tool calls
- LLM responses were unparseable by Ollama

**Fix**: Renamed method from `inject_initial_context(self, context)` to `on_initial_context(self, agent, context)`

## Bug #2: Missing qwen3-coder XML Format Support

**Location**: `behaviors/tool_calling_syntax.py:240-298`

**Problem**: XML parser only supported Anthropic format but qwen3-coder uses different syntax:
- Anthropic: `<invoke name="tool_name"><parameter name="arg">value</parameter></invoke>`
- qwen3-coder: `<function=tool_name><parameter=arg>value</parameter></function>`

**Impact**:
- Even when LLM returned XML tool calls, they weren't parsed
- Parser returned None, leaving tool_calls empty
- Agents saw no tool calls, made no progress

**Fix**: Added regex patterns to parse both XML formats with fallback

## Evidence

### Before Fix - Empty Context Injection

Context inspection file `.context_inspection/orchestrator_round_001_pre_llm.json` showed:

```
Position 0: System prompt
Position 1: "CHAT MODE: Answer questions..." (from ChatbotBehavior)
Position 2: "EXECUTION MODE..." (from ExecutionModeBehavior)
Position 3: "Available Agents..." (from DelegationBehavior)
Position 4: Goal message
Position 5: Execution mode reminder
```

**JSON examples were MISSING** - should have been at position 1.

### After Fix - Correct Injection

Context inspection file shows:

```
Position 0: System prompt
Position 1: JSON tool calling format examples ✓
Position 2: Chat mode message
Position 3: Execution mode message
Position 4: Tool calling format (from ToolCallingSyntaxBehavior) ✓
...
```

## Evaluation Results

### Before Fixes

```
Success rate: 0/4 (0.0%)
All tasks: Empty workspace, no files created
Pattern: 21+ consecutive empty rounds, then auto-fail
```

### After Fixes

Re-running evaluation now...

## Commit

Commit: `b90e0ef`
Message: "fix: Fix ToolCallingSyntaxBehavior event hook name and add qwen3-coder XML format support"

## Related Documents

- `/workspace/CURRENT_STATUS.md` - Original fix attempt (incomplete)
- `/workspace/docs/TOOL_CALLING_AND_ERROR_FEEDBACK_PLAN.md` - Implementation plan
- `/workspace/evaluation_results/l3_l7_failure_analysis.md` - Failure analysis
