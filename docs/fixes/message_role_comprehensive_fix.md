# Comprehensive Message Role Fix

## Problem

All framework instructions use `role="user"` when they should use `role="system"`. This includes:
- Tool documentation (injected via `inject_user_message_after_system`)
- Nudges/warnings (appended via direct `context.append()`)
- Mode instructions (both inject and append)

## All Context Modification Points

### 1. `inject_user_message_after_system()` - Used by 15+ behaviors
**Purpose:** Inject framework instructions after system prompt
**Current:** `context.insert(1, {"role": "user", ...})`
**Should be:** `context.insert(1, {"role": "system", ...})` (default)

**Used by:**
- All tool documentation (`on_initial_context`)
- Delegation info
- Tool calling format example

### 2. Direct `context.append()` - Used by 5 behaviors
**Purpose:** Append nudges/warnings at end of context
**Current:** `context.append({"role": "user", ...})`
**Should be:** `context.append({"role": "system", ...})` (default)

**Used by:**
- `chatbot.py:423` - Chat instructions (SHOULD STAY USER - actual chat mode)
- `execution_mode.py:269` - Execution mode nudge (SHOULD BE SYSTEM)
- `loop_detection.py:243` - Loop warnings (SHOULD BE SYSTEM)
- `time_box.py:155` - Time nudge (SHOULD BE SYSTEM)
- `time_box.py:169` - Custom reminder (SHOULD BE SYSTEM)

### 3. Direct `context.insert()` - Used by 1 behavior
**Purpose:** Insert format example after system prompt
**Current:** `context.insert(1, {"role": "user", ...})`
**Should be:** `context.insert(1, {"role": "system", ...})`

**Used by:**
- `tool_calling_syntax.py:86` - Tool calling format (SHOULD BE SYSTEM)

## Solution: Two Helper Methods

Add two helper methods in `behaviors/base.py` to centralize all context modifications:

```python
# In behaviors/base.py

def inject_message_after_system(
    self,
    context: list[dict[str, Any]],
    message: str,
    role: str = "system"
) -> list[dict[str, Any]]:
    """
    Inject a message after the system prompt (position 1).

    Use for framework instructions that need to appear early in context
    (tool docs, delegation info, format examples).

    Args:
        context: Current context (list of message dicts)
        message: Message content to inject
        role: Message role - "system" for framework (default),
              "user" for actual user messages

    Returns:
        Modified context with message injected

    Examples:
        # Tool documentation (system - default)
        self.inject_message_after_system(context, tool_docs)

        # Goal from user (user - explicit)
        self.inject_message_after_system(context, f"GOAL: {goal}", role="user")
    """
    if len(context) > 0:
        context.insert(1, {
            "role": role,
            "content": message
        })
    return context


def append_message(
    self,
    context: list[dict[str, Any]],
    message: str,
    role: str = "system"
) -> list[dict[str, Any]]:
    """
    Append a message to the end of context.

    Use for nudges, warnings, and reminders that should appear at the
    end of context (near where the issue is occurring).

    Args:
        context: Current context
        message: Message content to append
        role: Message role - "system" for framework nudges (default),
              "user" for actual user messages

    Returns:
        Modified context with message appended

    Examples:
        # Loop warning (system - default)
        self.append_message(context, loop_warning)

        # Chat mode instructions (user - explicit)
        self.append_message(context, chat_instructions, role="user")
    """
    context.append({
        "role": role,
        "content": message
    })
    return context
```

## Migration Plan

### Phase 1: Add Helper Methods (1 file)

✅ **behaviors/base.py**
- Add `inject_message_after_system()` with `role: str = "system"`
- Add `append_message()` with `role: str = "system"`
- Mark `inject_user_message_after_system()` as deprecated (keep for backward compat)

### Phase 2: Update Behaviors Using Helpers (18+ files)

**No changes needed** - All behaviors using `inject_user_message_after_system()` automatically get system role:
- All tool documentation behaviors (10+ files) ✅
- Delegation behavior ✅
- Validation behavior ✅

**Needs explicit `role="user"`** (2 behaviors):
- Find any goal/task injection that uses the helper

### Phase 3: Update Direct context.append() Uses (5 files)

**chatbot.py:423** - Chat instructions
```python
# BEFORE:
context.append({
    "role": "user",
    "content": chat_instructions
})

# AFTER:
self.append_message(context, chat_instructions, role="user")  # Explicit user
```

**execution_mode.py:269** - Execution nudge
```python
# BEFORE:
context.append({
    "role": "user",
    "content": self.pending_nudge
})

# AFTER:
self.append_message(context, self.pending_nudge)  # Default system
```

**loop_detection.py:243** - Loop warnings
```python
# BEFORE:
context.append({
    "role": "user",
    "content": warning_text
})

# AFTER:
self.append_message(context, warning_text)  # Default system
```

**time_box.py:155 and :169** - Time nudges
```python
# BEFORE:
nudge_message = {"role": "user", "content": msg}
context.append(nudge_message)

# AFTER:
self.append_message(context, msg)  # Default system
```

### Phase 4: Update Direct context.insert() Uses (1 file)

**tool_calling_syntax.py:86** - Format example
```python
# BEFORE:
context.insert(1, {
    "role": "user",
    "content": example
})

# AFTER:
self.inject_message_after_system(context, example)  # Default system
```

## Files to Update

**Core helpers (1 file):**
- ✅ `behaviors/base.py` - Add two new helpers

**Behaviors with direct context modifications (6 files):**
- ⚠️ `behaviors/chatbot.py` - Use `append_message(..., role="user")`
- ✅ `behaviors/execution_mode.py` - Use `append_message(...)` (default system)
- ✅ `behaviors/loop_detection.py` - Use `append_message(...)` (default system)
- ✅ `behaviors/time_box.py` - Use `append_message(...)` twice (default system)
- ✅ `behaviors/tool_calling_syntax.py` - Use `inject_message_after_system(...)` (default system)

**No changes needed (18+ files):**
- All behaviors using `inject_user_message_after_system()` automatically fixed ✅

**Total effort:** 7 files to update

## Testing Strategy

1. Add helper methods to `behaviors/base.py`
2. Update 1-2 high-impact behaviors (execution_mode, loop_detection)
3. Run quick L5 eval (1-2 tasks) to verify no regression
4. Update remaining behaviors
5. Full L5-L7 regression test

## Expected Impact

**Benefits:**
- ✅ Proper LLM understanding (system vs user context)
- ✅ Better token efficiency (some implementations)
- ✅ Aligned with OpenAI/Anthropic best practices
- ✅ Centralized context modification (easier to maintain)
- ✅ Explicit role override for user messages (safer)

**Risks:**
- ⚠️ Behavior change - LLM might treat messages differently
- ⚠️ Need thorough testing

## Summary

**Current state:**
- All framework instructions incorrectly use `role="user"`
- Mix of helper usage and direct context modification
- No centralized way to append messages

**Proposed state:**
- Two centralized helpers with `role="system"` default
- Explicit `role="user"` override for actual user messages
- Consistent pattern across all behaviors

**Migration:**
- 1 file: Add helpers
- 6 files: Update to use helpers
- 18+ files: Automatically fixed (no changes)
