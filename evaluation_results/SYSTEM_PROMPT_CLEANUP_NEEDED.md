# System Prompt Cleanup: Remove Hardcoded Tool References

**Date:** 2025-11-08
**Issue:** System prompts hardcode tool names, violating behavior composability principle
**Impact:** Breaks when behaviors are added/removed, reduces modularity

---

## The Problem

System prompts currently reference specific tools by name (e.g., "use list_dir", "call mark_complete"). This violates the behavior system's composability principle:

**Behaviors should inject tool documentation dynamically.**
**System prompts should describe WHAT to do, not HOW (which tools).**

### Why This Matters

1. **Breaks composability**: If you remove a behavior, the prompt still references its tools
2. **Maintenance burden**: Adding/removing behaviors requires updating multiple prompts
3. **Inconsistency**: Tools available don't match what prompt describes
4. **Misleading agents**: Prompts reference tools that might not exist

---

## Violations Found

### 1. task_executor.yaml (Lines 62, 71-73)

**Current (BAD):**
```yaml
IMPORTANT - UNDERSTAND THE WORKSPACE FIRST:
**Before starting implementation, understand what's already in the workspace**:
1. Use list_dir to see the workspace structure (check for existing code, architecture docs, tests, etc.)
2. If you find existing files, read them to understand the current state
...
- ALWAYS use list_dir and read_file to verify what files actually exist
- NEVER call mark_complete based solely on what workspace_task_notes.md says
```

**Should be (GOOD):**
```yaml
IMPORTANT - UNDERSTAND THE WORKSPACE FIRST:
**Before starting implementation, understand what's already in the workspace**:
1. Inspect the workspace structure to check for existing code, architecture docs, tests, etc.
2. If you find existing files, examine them to understand the current state
...
- ALWAYS verify what files actually exist before proceeding
- NEVER rely solely on summaries or notes - check the actual workspace state
- Signal completion only when the goal is fully achieved
```

**Rationale**: Describes the PROCESS (inspect, verify, signal completion) without naming tools (list_dir, read_file, mark_complete). Behaviors inject the actual tools.

---

### 2. architect.yaml (Lines 49-56)

**Current (BAD):**
```yaml
YOUR TOOLS (ONLY THESE - NOTHING ELSE):
✅ write_architecture_doc - Create system design overview
✅ write_module_spec - Document individual modules/components
✅ write_task_list - Break down implementation tasks
✅ mark_complete - Signal architecture documentation is complete
✅ mark_failed - Signal if architecture cannot be designed

❌ You DO NOT have: write_file, read_file, run_bash, or ANY implementation tools
❌ You CANNOT create code files (app.py, test_*.py, requirements.txt, etc.)
❌ You CANNOT run commands or execute tests
```

**Should be (GOOD):**
```yaml
🚨 CRITICAL: YOU ARE A DESIGN-ONLY AGENT 🚨

YOUR ROLE: Create architecture documentation (system design, module specs, task lists)
NOT YOUR ROLE: Implement code, write application files, run tests, execute commands

You specialize in:
✅ Architecture documentation and design artifacts
✅ System decomposition and module specifications
✅ Task breakdown and implementation planning
✅ Signaling when design work is complete

You do NOT implement:
❌ Application code files (*.py, *.js, *.html, etc.)
❌ Test files or test execution
❌ Command execution or validation

IF YOU CANNOT COMPLETE A DESIGN TASK:
→ Signal failure with a clear reason
→ DO NOT keep trying or explaining why you can't proceed
→ DO NOT have empty rounds - always make progress or signal completion/failure
```

**Rationale**: Describes ROLE and CAPABILITIES without listing specific tool names. The architect tools behavior will inject the actual tool documentation. The prohibition list describes FILE TYPES and CAPABILITIES, not tool names.

---

### 3. meta_programmer.yaml (Lines 66, 69-73)

**Current (BAD):**
```yaml
1. **Understand the requirement**:
   - Ask clarifying questions if the task is vague
   - Identify what tools, context, or hooks are needed
   - Check for existing similar behaviors (use list_dir, read_file)

2. **Read templates first**:
   ```
   read_file("behaviors/templates/behavior_minimal_template.py")
   read_file("behaviors/templates/behavior_with_tools_template.py")
   read_file("behaviors/templates/behavior_test_template.py")
   ```
```

**Should be (GOOD):**
```yaml
1. **Understand the requirement**:
   - Ask clarifying questions if the task is vague
   - Identify what tools, context, or hooks are needed
   - Check for existing similar behaviors in the codebase

2. **Read templates first**:
   - Examine the minimal behavior template
   - Review the behavior-with-tools template
   - Study the test template
   - Templates are in: behaviors/templates/
```

**Rationale**: Describes WHAT to do (check for existing, read templates) without HOW (list_dir, read_file). The file operation behaviors will inject the actual tools. We provide PATHS but not TOOL NAMES.

---

### 4. task_executor_with_inspection.yaml (Line 35)

**Current (BAD):**
```yaml
Work systematically:
1. Plan your approach
2. Implement incrementally
3. Test thoroughly
4. Fix any issues
5. Call mark_complete() when done
```

**Should be (GOOD):**
```yaml
Work systematically:
1. Plan your approach
2. Implement incrementally
3. Test thoroughly
4. Fix any issues
5. Signal completion when the goal is fully achieved
```

**Rationale**: "Signal completion" instead of "Call mark_complete()". The actual tool name is injected by behaviors.

---

## General Principles for System Prompts

### DO: Describe Processes and Workflows

**Good examples:**
- "Inspect the workspace structure before starting"
- "Verify files exist before proceeding"
- "Signal completion when the goal is achieved"
- "Create architecture documentation for the system"
- "Break down the implementation into structured tasks"
- "Generate code following template patterns"

### DON'T: Reference Specific Tools

**Bad examples:**
- "Use list_dir to see files" ❌
- "Call mark_complete when done" ❌
- "Use write_file to create files" ❌
- "Run read_file to check contents" ❌
- "Available tools: X, Y, Z" ❌

### DO: Describe Capabilities and Roles

**Good examples:**
- "You can inspect directory structures"
- "You have file manipulation capabilities"
- "You can execute shell commands"
- "You specialize in architecture design"
- "You do NOT implement code - design only"

### DON'T: List Tool Names or Signatures

**Bad examples:**
- "write_architecture_doc(path, content)" ❌
- "Tools: list_dir, read_file, write_file" ❌
- "You have: mark_complete, mark_failed" ❌

### DO: Provide File Paths or Locations

**Good examples:**
- "Templates are in: behaviors/templates/"
- "Architecture docs in: architecture/"
- "Check workspace_task_notes.md for summaries"

These are DATA locations, not tool references.

### DON'T: Show Example Tool Calls

**Bad examples:**
- `read_file("behaviors/templates/template.py")` ❌
- `list_dir("/workspace")` ❌

Instead, say "Examine template files" or "Inspect workspace".

---

## Implementation Plan

### Phase 1: Fix Critical Configs (Blocks Evaluation)

1. **task_executor_with_inspection.yaml** - Used for L4-L7 eval
   - Line 35: "Signal completion when goal achieved"

### Phase 2: Fix Core Agent Configs

2. **task_executor.yaml** - Most commonly used agent
   - Lines 62, 71-73: Genericize workspace inspection instructions
   - Remove all "use X tool" references

3. **architect.yaml** - Used by orchestrator
   - Lines 49-56: Describe capabilities, not tools
   - Focus on role/responsibilities distinction

4. **meta_programmer.yaml** - Used for extensibility
   - Lines 66, 69-73: Describe workflow, not tool calls
   - Keep file paths, remove tool names

### Phase 3: Audit All Other Configs

5. Check orchestrator.yaml, any other agent configs
6. Verify no other prompts reference tool names

---

## Testing Strategy

After fixes:

1. **Behavior removal test**: Remove a behavior from config, verify prompt still makes sense
2. **Behavior addition test**: Add a new behavior, verify it works without prompt changes
3. **Cross-agent test**: Verify agents work with different behavior combinations

---

## Benefits After Cleanup

1. **True composability**: Add/remove behaviors without touching prompts
2. **Clear separation**: Prompts = WHAT to do, Behaviors = HOW to do it
3. **Easier maintenance**: One place to update tool docs (the behavior)
4. **Better modularity**: Behaviors self-contained and swappable
5. **No surprises**: Agents never reference tools they don't have

---

## Example: Before and After

### BEFORE (Bad - Hardcoded Tools):

```yaml
system_prompt: |
  You are a coding agent.

  Workflow:
  1. Use list_dir to see files
  2. Use read_file to check existing code
  3. Use write_file to create new files
  4. Use run_bash to run tests
  5. Call mark_complete when done

  Available tools:
  - list_dir(path): List directory contents
  - read_file(path): Read file
  - write_file(path, content): Write file
  - run_bash(cmd): Execute command
  - mark_complete(): Finish task

behaviors:
  - type: DirectoryToolsBehavior
  - type: ReadFileToolsBehavior
  - type: WriteFileToolsBehavior
  - type: CommandToolsBehavior
```

**Problems:**
- Prompt lists exact tools
- If you remove WriteFileToolsBehavior, prompt still says "use write_file"
- Maintenance nightmare - every behavior change requires prompt update

### AFTER (Good - Process-Focused):

```yaml
system_prompt: |
  You are a coding agent that implements software projects.

  Workflow:
  1. Inspect the workspace to understand existing structure
  2. Examine existing code to understand current state
  3. Create or modify files as needed for the implementation
  4. Validate your work by running appropriate tests
  5. Signal completion when the goal is fully achieved

  You have capabilities for:
  - Workspace inspection and file navigation
  - File reading and content examination
  - File creation and modification
  - Command execution for testing and validation
  - Task completion signaling

behaviors:
  - type: DirectoryToolsBehavior      # Injects list_dir tool + docs
  - type: ReadFileToolsBehavior        # Injects read_file tool + docs
  - type: WriteFileToolsBehavior       # Injects write_file tool + docs
  - type: CommandToolsBehavior         # Injects run_bash tool + docs
```

**Benefits:**
- Prompt describes PROCESS, not TOOLS
- Remove WriteFileToolsBehavior? Prompt still makes sense (just can't create files)
- Add new behavior? No prompt change needed
- Each behavior injects its own tool documentation dynamically
- True separation of concerns

---

## Conclusion

System prompts should be **process-focused and behavior-agnostic**.

**Rule of thumb:**
If your system prompt wouldn't make sense with a different set of behaviors, you're doing it wrong.

**Test:**
Could you swap out behaviors and the prompt still describes a coherent workflow? If yes, good. If no, you've hardcoded tool names.
