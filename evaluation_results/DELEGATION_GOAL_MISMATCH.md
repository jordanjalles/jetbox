# Delegation Goal Mismatch - Architect Role Confusion

**Date**: 2025-11-02
**Issue**: Architect has empty rounds because of implementation-focused delegation goals
**Status**: ROOT CAUSE IDENTIFIED

## Problem

Orchestrator delegates to architect with implementation-focused goals:
```
[delegation] Delegating to architect: Full-stack Flask application with user authentication, posts...
```

Architect interprets this as "I need to implement this application" but only has architecture tools:
```
Available tools:
  - write_architecture_doc
  - write_module_spec
  - write_task_list
  - list_architecture_docs
  - read_architecture_doc
```

LLM response:
```
"I'm unable to create the required Flask application files because the
available toolset does not include a file‑creation utility (e.g., `write_file`)."
```

Result: 50 consecutive empty rounds, architect fails.

## Root Cause

**Mismatch between delegation goal and architect role:**

1. **Orchestrator's delegation goal**: "Full-stack Flask application..." (sounds like implementation)
2. **Architect's actual role**: "Design architecture, create documentation" (not implementation)
3. **LLM confusion**: Goal sounds like implementation, but tools are for documentation
4. **Result**: LLM doesn't know what to do, produces empty rounds

## Expected vs Actual

### Expected Delegation Goal (for architect)

```
"Design the architecture for a full-stack Flask application with user
authentication, posts, and comments. Create architecture documents, module
specifications, and task breakdown."
```

### Current Delegation Goal (problematic)

```
"Full-stack Flask application with user authentication, posts, and comments.
Uses SQLite database. Includes frontend templates using Jinja2 and Bootstrap 5."
```

The current goal sounds like a work order for implementation, not a consultation request.

## Why ChatbotBehavior Fix Didn't Solve This

The ChatbotBehavior fix prevented chat mode activation (✅ working), but revealed a different issue:

**Before fix**: Architect entered chat mode, asked clarifying questions
**After fix**: Architect stayed in execution mode, but confused about what to execute

The architect now correctly recognizes it has a goal and should execute, but the goal description makes it think it should write code files, not architecture documents.

## Solutions

### Option 1: Fix Orchestrator Delegation Prompts (Recommended)

Update orchestrator's system prompt or delegation behavior to phrase architect goals correctly:

**Current**:
```python
goal = f"{user_request}"  # e.g., "Create a Flask app..."
```

**Fixed**:
```python
# When delegating to architect
if delegating_to == "architect":
    goal = f"Design the architecture for: {user_request}. Create architecture documents, module specs, and task breakdown."
```

### Option 2: Update Architect System Prompt Clarity

Make the architect's role even more explicit:

```yaml
system_prompt: |
  You are an Architecture Consultant, NOT an implementation agent.

  IMPORTANT - YOUR ROLE:
  - You DESIGN architecture (high-level)
  - You CREATE documentation (architecture docs, module specs, task lists)
  - You DO NOT write code files (no .py, .js, .html files)
  - You DO NOT implement features

  When given an implementation task like "Create a Flask app", interpret it as:
  "Design the architecture FOR a Flask app" and use your architecture tools.
```

### Option 3: Add Tool Guidance in Architect Prompt

Add explicit tool usage reminders:

```yaml
## Your Tools

You have specialized architecture tools:
- write_architecture_doc: Create system design documents
- write_module_spec: Document individual modules/components
- write_task_list: Break down into implementation tasks

You do NOT have code writing tools. That's by design - you're a consultant, not a coder.

If a goal sounds like "build X", interpret it as "design the architecture for X" and use your tools.
```

## Recommendation

**Implement all 3 solutions**:

1. ✅ **Option 1** (highest priority): Fix orchestrator delegation phrasing
   - Impact: Prevents confusion at the source
   - Effort: Update delegation behavior prompt

2. ✅ **Option 2** (medium priority): Clarify architect role in system prompt
   - Impact: Handles edge cases where goal is ambiguous
   - Effort: Update architect_config.yaml

3. ✅ **Option 3** (low priority): Add tool guidance
   - Impact: Extra safety net
   - Effort: Update architect_config.yaml

## Next Steps

1. Update orchestrator's delegation behavior to phrase architect goals as "Design architecture for..."
2. Update architect system prompt to explicitly state it's NOT a coder
3. Rerun L7 evaluation to verify fix

## Related Issues

- Empty rounds root cause (FIXED): ChatbotBehavior not detecting SubAgentModeBehavior's goal
- This issue: Goal phrasing mismatch between orchestrator and architect roles
