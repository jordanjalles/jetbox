# Message Role Fix: System vs User

## Problem

Framework instructions are currently being added as `role="user"` messages when they should be `role="system"` messages. This wastes context tokens and weakens the LLM's understanding of what is system guidance vs user requests.

**Evidence from context snapshot:**
```
0: role=system     | # CONTEXT You are a software architecture consultant...
1: role=user       | architect_tools tools: ... [SHOULD BE SYSTEM]
2: role=user       | CHAT MODE: Answer questions conversationally... [SHOULD BE SYSTEM]
3: role=user       | EXECUTION MODE: You must call at least one tool... [SHOULD BE SYSTEM]
4: role=user       | ## Tool Calling Format ... [SHOULD BE SYSTEM]
5: role=user       | GOAL: Create a blog system... [This one could be user or system]
6: role=user       | 🔧 EXECUTION MODE: You must call at least one tool... [SHOULD BE SYSTEM]
```

## Root Cause

**`behaviors/base.py:576-579`** - `inject_user_message_after_system()` hardcodes `"role": "user"`:

```python
def inject_user_message_after_system(
    self,
    context: list[dict[str, Any]],
    message: str
) -> list[dict[str, Any]]:
    """Helper to inject a user message after the system prompt (index 1)."""
    if len(context) > 0:
        context.insert(1, {
            "role": "user",      # ❌ HARDCODED AS USER
            "content": message
        })
    return context
```

**Used by 15+ behaviors:**
- `WriteFileToolsBehavior.on_initial_context()` - Tool documentation
- `ReadFileToolsBehavior.on_initial_context()` - Tool documentation
- `CommandToolsBehavior.on_initial_context()` - Tool documentation
- `ExecutionModeBehavior.on_initial_context()` - Mode instructions
- `ExecutionModeBehavior.on_round_start()` - Mode nudges
- `ToolCallingSyntaxBehavior.on_initial_context()` - Format examples
- `DelegationBehavior.on_initial_context()` - Delegation docs
- `LoopDetectionBehavior.on_round_start()` - Loop warnings
- And 7+ others...

## Solution

### Recommended: Add Role Parameter (Default to System)

Modify existing helper to accept optional role parameter, **defaulting to "system"** since most uses are framework instructions:

```python
# In behaviors/base.py

def inject_user_message_after_system(
    self,
    context: list[dict[str, Any]],
    message: str,
    role: str = "system"  # ✅ Default to system (framework instructions)
) -> list[dict[str, Any]]:
    """
    Helper to inject a message after the system prompt (index 1).

    Use this for framework instructions (tool docs, mode instructions, nudges)
    and user-facing messages (goals, task assignments).

    Args:
        context: Current context (list of message dicts)
        message: Message content to inject
        role: Message role - "system" for framework instructions (default),
              "user" for actual user messages/goals

    Returns:
        Modified context with message injected at index 1

    Examples:
        # Tool documentation (system)
        self.inject_user_message_after_system(context, tool_docs)

        # Goal from orchestrator (user)
        self.inject_user_message_after_system(context, f"GOAL: {goal}", role="user")
    """
    if len(context) > 0:
        context.insert(1, {
            "role": role,
            "content": message
        })
    return context
```

**Why default to "system"?**
- 95% of current uses are framework instructions (tool docs, modes, nudges)
- Only 5% are actual user messages (goals from orchestrator)
- Safer default - framework instructions should be system
- Explicit override needed for user messages (more intentional)

**Migration:**

1. **Update helper signature** - Add `role: str = "system"` parameter
2. **No changes needed for framework instructions** - They default to system now ✅
3. **Explicit override for user messages** - Only goal/task assignments need `role="user"`

**Files needing explicit `role="user"`:**
- `behaviors/delegation.py` - Goal injection for delegated tasks
- Orchestrator goal injection (if it uses this helper)
- Any user-facing task descriptions

**All other uses automatically become system:**
- Tool documentation (10+ behaviors) ✅
- Mode instructions (ExecutionModeBehavior) ✅
- Nudges (LoopDetection, TimeBox, etc.) ✅
- Tool calling format ✅

## Files Requiring Updates

**Core helper (1 file) - MUST UPDATE:**
- ✅ `behaviors/base.py` - Add `role: str = "system"` parameter to `inject_user_message_after_system()`

**Files needing explicit `role="user"` (3-5 files):**
- ⚠️ `behaviors/delegation.py` - Goal injection for delegated tasks
- ⚠️ `src/agent_lifecycle.py` - Initial goal injection (if it uses this helper)
- ⚠️ Check for orchestrator goal propagation

**No changes needed (automatically become system):**
- ✅ All tool documentation behaviors (10+ files)
- ✅ Mode instruction behaviors (3 files)
- ✅ Warning/nudge behaviors (5+ files)
- ✅ Total: ~18 behaviors automatically fixed

## Impact Assessment

**Benefits:**
- ✅ Clearer LLM context (system vs user)
- ✅ Potentially better token efficiency (some LLM implementations)
- ✅ More aligned with OpenAI/Anthropic best practices
- ✅ Easier to debug (clear role boundaries)

**Risks:**
- ⚠️ Behavior change - LLM might treat system messages differently
- ⚠️ Need to test thoroughly across all behaviors
- ⚠️ ~20 files to update

**Testing Strategy:**
1. Add new helper method
2. Update 1-2 behaviors (e.g., WriteFileToolsBehavior, ExecutionModeBehavior)
3. Run L5 evaluation to verify no regression
4. If successful, migrate remaining behaviors
5. Re-run full L5-L7 evaluation

## Next Steps

1. Implement `inject_system_message_after_prompt()` in `behaviors/base.py`
2. Update ExecutionModeBehavior (high impact, easy to test)
3. Run quick eval on 1-2 L5 tasks
4. If successful, create migration script for remaining behaviors
5. Full regression testing

## Notes

- Goal messages from orchestrator might stay as `role="user"` (debatable)
- Tool results from `dispatch_tool()` should remain `role="user"` (actual LLM output)
- Only framework-injected instructions should change to `role="system"`
