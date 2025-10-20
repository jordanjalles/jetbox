# Experiment Results: Testing 3B Model with Agent

## Experiment Setup

**Goal**: Test if qwen2.5-coder:3b can run the agent while gaming

**Task Given**: "Create mathx package with add(a,b) and multiply(a,b) functions. Add tests for both. Run ruff and pytest."

## Results

### ❌ Qwen Models Don't Support Ollama Tool Calling

**Discovery**: Both qwen2.5-coder:3b and qwen2.5-coder:7b do NOT properly support Ollama's tool calling format.

#### What Happened:

1. **3b Model Test**:
   - Agent probed state (ran ruff, pytest - both failed as expected on empty dir)
   - Model was called with tool specifications
   - **Model returned text response instead of tool_calls**
   - Agent stopped after Round 1 (no tools to execute)

2. **7b Model Test**:
   - Same behavior as 3b
   - Model understands tools but returns JSON in `content` field
   - Does not use Ollama's `tool_calls` structure

3. **Verification Test**:
```python
# Qwen response format:
{
  "message": {
    "role": "assistant",
    "content": '{"name": "write_file", "arguments": {...}}',  # ❌ Wrong format
    "tool_calls": None  # ❌ Should be here
  }
}

# GPT-OSS:20b response format (correct):
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [...]  # ✓ Proper Ollama format
  }
}
```

#### Why This Matters:

The `agent.py` expects Ollama's standard tool calling format where tools are returned in the `tool_calls` field. Qwen models return tool information as JSON text in the `content` field, which the agent doesn't parse.

## Conclusion

**Qwen models (both 3b and 7b) are NOT compatible with this agent** because they don't follow Ollama's tool calling specification.

## Alternative Solutions

### Option 1: Stick with gpt-oss:20b (Current)
- **Pros**: Works perfectly with tools
- **Cons**: Heavy (13 GB), slower
- **Recommendation**: Use when NOT gaming

### Option 2: Find Other Lightweight Models
Models to try that MAY support Ollama tool calling:
- **llama3.2:3b** - Meta's model, might support tools
- **phi3.5:3.8b** - Microsoft's model
- **gemma2:2b** - Google's model

### Option 3: Modify Agent for Qwen
Would require:
1. Parsing JSON from `content` field when `tool_calls` is None
2. Converting parsed JSON to tool call format
3. Testing with both model types

**Complexity**: Medium
**Time**: 1-2 hours of development

## Performance Data

| Model | Size | Tool Support | Speed | Result |
|-------|------|--------------|-------|--------|
| qwen2.5-coder:3b | 1.9 GB | ❌ No (wrong format) | ⚡⚡⚡ Fast | Failed |
| qwen2.5-coder:7b | 4.7 GB | ❌ No (wrong format) | ⚡⚡ Fast | Failed |
| gpt-oss:20b | 13 GB | ✅ Yes | ⚡ Moderate | Works |

## Recommendations

### For Gaming Sessions:

**Short term**: Don't run the agent while gaming with current setup. Use `gpt-oss:20b` when NOT gaming.

**Medium term**: Try alternative lightweight models (llama3.2, phi3.5, gemma2) to see if they support Ollama tool calling.

**Long term**: Modify agent to support Qwen's JSON-in-content tool format, OR wait for Qwen to update Ollama support.

## Next Steps

1. ✅ Documented findings
2. ⏭ Test llama3.2:3b or phi3.5:3.8b for tool calling support
3. ⏭ If those fail, consider modifying agent to parse Qwen's format
4. ⏭ Or accept using gpt-oss:20b only when not gaming

## Code Snippets

### Test Tool Calling Support:
```python
from ollama import chat

resp = chat(
    model='MODEL_NAME_HERE',
    messages=[{'role': 'user', 'content': 'Write hello to test.txt'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': 'Write a file',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'}
                },
                'required': ['path', 'content']
            }
        }
    }]
)

print('Tool calls:', resp['message'].get('tool_calls'))
print('Content:', resp['message'].get('content'))
```

### Expected Output for Compatible Model:
```
Tool calls: [{'function': {'name': 'write_file', 'arguments': {...}}}]
Content:
```

### Actual Output for Qwen:
```
Tool calls: None
Content: {"name": "write_file", "arguments": {...}}
```

## Lessons Learned

1. **Not all Ollama models support tool calling** - even if they're good at coding
2. **Tool calling format varies** - Qwen uses a different approach than Ollama standard
3. **Model size != capability** - 3b and 7b both failed in the same way
4. **Always test with simple tasks first** before assuming compatibility

---

**Date**: October 19, 2025
**Models Tested**: qwen2.5-coder:3b, qwen2.5-coder:7b, gpt-oss:20b
**Agent Version**: agent.py (Ollama tool calling format)
