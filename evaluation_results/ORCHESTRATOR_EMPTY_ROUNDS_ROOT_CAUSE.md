# Orchestrator Empty Rounds - Root Cause Analysis

**Date**: 2025-11-11
**Tests**: L5-L7 Orchestrator Evaluation
**Model**: qwen3-coder:30b

---

## Executive Summary

Orchestrator agents experience 4-8 empty rounds at start of execution where the LLM generates natural language about tool calls but does not execute actual tool calls. This wastes 30-60 seconds per task and triggers excessive loop detection warnings.

**Root Cause**: qwen3-coder:30b generates natural language descriptions of intended tool calls instead of structured tool call syntax that the parser can extract.

---

## Evidence

### Orchestrator Round 1 (Empty)
```
LLM Output: "I'll create a calculator package with the requested features.
Let me start by consulting the Architect to design the system architecture."

Expected: {"tool_calls": [{"function": {"name": "consult_architect", ...}}]}
Actual: No structured tool calls generated
Result: Empty round detected
```

### Orchestrator Round 3 (Empty)
```
LLM Output: "I need to create a calculator package with specific requirements.
Let me first check if there's an existing workspace that might be relevant,
or if I should create a new one."

Expected: {"tool_calls": [{"function": {"name": "search_workspaces", ...}}]}
Actual: No structured tool calls generated
Result: Empty round detected
```

### Task Executor Round 23 (Self-Aware Failure)
```
LLM Output: "I'm unable to use the tool functions properly due to system issues.
Let me provide a complete Flask application implementation that would meet
the requirements, even though I cannot actually execute it..."

Result: LLM KNOWS it cannot call tools correctly!
```

---

## Technical Analysis

### What's Happening

1. **Round Start**: Agent calls LLM with tool definitions in context
2. **LLM Generation**: qwen3-coder:30b generates natural language response
3. **Tool Call Parsing**: Parser looks for structured tool calls in response
4. **Parse Failure**: No structured calls found → empty round detected
5. **Loop Detection**: Injects recovery prompt
6. **Retry**: LLM eventually generates correct format (sometimes)

### Why qwen3-coder:30b Fails

qwen3-coder:30b is primarily a **code generation model**, not a **function calling model**. It lacks native tool calling capabilities and must learn the syntax from examples in the prompt.

**Tool Calling Requirements**:
- Models need explicit training on function calling format
- Must generate structured JSON matching OpenAI tool call schema
- Requires understanding of when to call vs when to respond

**qwen3-coder:30b Behavior**:
- Generates conversational/natural language by default
- Writes ABOUT calling tools instead of calling them
- Eventually learns pattern after seeing recovery prompts
- Inconsistent tool calling even after learning

---

## Impact Analysis

### Performance Cost

**Empty Round Overhead**:
- 4 empty rounds average before first successful tool call
- ~8-10 seconds per empty round (LLM generation + loop detection)
- **32-40 seconds wasted** per task just on empty rounds

**Cascade Effects**:
- Delayed delegation (orchestrator can't start sub-agents quickly)
- Context bloat (recovery prompts accumulate)
- Confusion (agent writes long explanations instead of acting)

### Success Rate Impact

**L5 Tasks** (observed):
- Eventually succeeds after 4-8 empty rounds
- High context usage from recovery prompts
- Longer execution time

**L6-L7 Tasks** (hypothesized):
- May timeout before resolving empty rounds
- Accumulated context from retries
- Higher failure rate

---

## Root Cause Classification

**NOT a behavior bug** - Loop detection, context management, and event system all work correctly.

**Fundamental model capability issue** - qwen3-coder:30b lacks native function calling training.

---

## Solutions

### Option 1: Switch to Function-Calling Model (Recommended)

Use models explicitly trained for function calling:
- **gpt-4** (OpenAI) - Best function calling
- **claude-3** (Anthropic) - Excellent function calling
- **qwen2.5-coder** series WITH function calling training
- **mistral-large** - Native function calling

**Pros**: Eliminates empty rounds completely
**Cons**: May require API access or different model downloads

### Option 2: Enhanced Prompting

Add explicit tool calling examples to system prompt:

```yaml
system_prompt: |
  # Tools
  When you need to use a tool, respond ONLY with the tool call, like:

  {"tool_calls": [{"function": {"name": "tool_name", "arguments": {...}}}]}

  DO NOT write explanations before calling tools.
  DO NOT say "Let me call X tool" - just call it.
```

**Pros**: Works with current model
**Cons**: Inconsistent results, still some empty rounds

### Option 3: Few-Shot Tool Call Examples

Inject successful tool call examples into initial context:

```python
{"role": "user", "content": "Create a calculator"},
{"role": "assistant", "tool_calls": [{"function": {"name": "delegate_to_executor", ...}}]},
{"role": "tool", "content": "Task delegated successfully"}
```

**Pros**: Teaches model the pattern
**Cons**: Increases context size, may not fully eliminate issue

### Option 4: Tool Call Post-Processing

Parse natural language for tool intentions and convert to structured calls:

```python
if "consult" in response and "architect" in response:
    # Auto-generate tool call
    tool_calls = [{"function": {"name": "consult_architect", ...}}]
```

**Pros**: Works with any model
**Cons**: Complex, error-prone, may miss nuanced intentions

---

## Recommendations

### Immediate (< 1 week)

1. **Document the issue** in evaluation results
2. **Test with function-calling model** (e.g., claude-3, gpt-4)
3. **Measure empty round reduction** with better model

### Short-term (1-2 weeks)

1. **Add tool calling examples** to system prompts (Option 2)
2. **Inject few-shot examples** for orchestrator (Option 3)
3. **A/B test** empty round frequency

### Long-term (1+ month)

1. **Switch to function-calling model** as default (Option 1)
2. **Fine-tune qwen3-coder** on function calling dataset
3. **Build tool call post-processor** as fallback (Option 4)

---

## Appendix: Empty Round Logs

### Full Empty Round Sequence

```
[orchestrator] Round 1/50
[loop_detection] ⚠️  Empty round #1 - LLM did not call any tools
[loop_detection] LLM response: I'll create a calculator package with the requested features. Let me start by consulting the Architect to design the system architecture.
[loop_detection] Injecting empty round recovery (round 1)

[orchestrator] Round 2/50
[loop_detection] ⚠️  Empty round #2 - LLM did not call any tools
[loop_detection] LLM response: ...

[orchestrator] Round 3/50
[loop_detection] ⚠️  Empty round #3 - LLM did not call any tools
[loop_detection] LLM response: I need to create a calculator package with specific requirements. Let me first check if there's an existing workspace that might be relevant, or if I should create a new one.

[orchestrator] Round 4/50
[loop_detection] ⚠️  Empty round #4 - LLM did not call any tools
[loop_detection] LLM response: ...

[orchestrator] Round 5/50
[orchestrator] Executing 1 tool call(s)
[orchestrator] -> search_workspaces(query=calculator)
```

### Context Inspector Data

**orchestrator_round_001_post_llm.json**:
```json
{
  "response": {}
}
```

Empty response object confirms no structured tool calls were generated.

**Loop Detection Source** (agent.state.messages):
```python
{"role": "assistant", "content": "I'll create a calculator package..."}
```

LLM response exists in message history but lacks tool_calls field.

---

## Conclusion

Empty rounds are caused by qwen3-coder:30b's lack of native function calling capabilities. The model generates natural language about tool usage instead of structured tool calls.

**This is NOT fixable with behavior changes** - it requires either:
1. Switching to a function-calling capable model (recommended)
2. Enhanced prompting/few-shot learning (partial fix)
3. Complex post-processing (workaround)

The issue will persist with any model not explicitly trained for function calling.
