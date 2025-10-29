# Cross-Subtask Context Passing - Design Proposals

## Problem Statement

Now that subtask isolation is working correctly (subtasks can't see each other's message history), we need a deliberate mechanism for passing important context between subtasks when necessary.

**Current state**:
- ✅ Subtasks properly isolated
- ✅ No implicit information leakage
- ⚠️ Only communication method: write/read files manually
- ❌ Agent must remember to write relevant info
- ❌ Agent must remember to read relevant info
- ❌ No standardized format

**Desired state**:
- ✅ Maintain subtask isolation
- ✅ Automatic/foolproof context passing when needed
- ✅ Agent doesn't have to "remember" to share
- ✅ Minimal cognitive load
- ✅ Clear, inspectable by humans

---

## Design Principles

1. **Explicit over implicit**: Better to have visible artifacts than hidden state
2. **Foolproof**: System should guide agent, not rely on agent memory
3. **Inspectable**: Humans should be able to see what's being shared
4. **Selective**: Not everything should carry over (defeats isolation purpose)
5. **Minimal friction**: Should feel natural to agent workflow

---

## Proposal 1: Auto-Summarization on Completion

### Mechanism
When agent calls `mark_subtask_complete(success=True)`, system immediately prompts:

```
System: "Before moving to the next subtask, summarize what you learned or discovered
that might be useful later. Keep it concise (2-3 key points)."

Agent: "Key findings:
- Project uses FastAPI framework
- Main entry point is server.py
- Database schema defined in models.py"

[System stores this in .agent_context/subtask_summaries.json]
```

On next subtask start, context includes:
```
PREVIOUS SUBTASK SUMMARIES:
• Subtask 1 (Analyze project structure):
  - Project uses FastAPI framework
  - Main entry point is server.py
  - Database schema defined in models.py
```

### Implementation
```python
# In mark_subtask_complete:
if success:
    # Prompt for summary
    summary_prompt = {
        "role": "system",
        "content": "Summarize 2-3 key findings from this subtask that future subtasks might need."
    }
    summary_response = call_llm([...context, summary_prompt])

    # Store summary
    subtask.summary = summary_response

    # On next subtask, include in context:
    all_summaries = [st.summary for st in completed_subtasks if st.summary]
```

### Pros
- ✅ Automatic - agent can't forget
- ✅ Concise - only key points
- ✅ Natural flow - happens at completion time
- ✅ Human-readable in state file

### Cons
- ⚠️ Extra LLM call per subtask (cost/latency)
- ⚠️ Agent might over-summarize or under-summarize
- ⚠️ Summary quality depends on agent understanding
- ⚠️ All summaries go to all subtasks (could be noisy)

### Variations
- **1a. Optional summaries**: Only prompt if subtask description contains keywords like "analyze", "investigate", "find"
- **1b. Structured summaries**: Use JSON format: `{"findings": [...], "decisions": [...], "resources": [...]}`
- **1c. Decay**: Only include summaries from last 3 subtasks to avoid clutter

---

## Proposal 2: Persistent Notes File

### Mechanism
System maintains `.agent_workspace/NOTES.md` that persists across subtasks.

**On subtask start**:
```
[Context includes]
SHARED NOTES (from NOTES.md):
## Project Structure
- FastAPI framework
- Entry: server.py

## Database
- Schema in models.py
- Uses PostgreSQL
```

**During subtask**:
Agent can call `update_notes(section, content)` tool:
```python
update_notes(
    section="API Endpoints",
    content="Found 3 main endpoints: /users, /posts, /auth"
)
```

**File automatically read on next subtask**.

### Implementation
```python
# New tool
def update_notes(section: str, content: str) -> dict:
    """Add or update a section in shared notes file."""
    notes_file = workspace / "NOTES.md"

    # Parse existing notes
    notes = parse_markdown(notes_file.read_text())

    # Update section
    notes[section] = content

    # Write back
    notes_file.write_text(format_markdown(notes))

    return {"status": "updated", "section": section}

# In build_context:
notes_file = workspace / "NOTES.md"
if notes_file.exists():
    context.append({
        "role": "user",
        "content": f"SHARED NOTES:\n{notes_file.read_text()}"
    })
```

### Pros
- ✅ Explicit file - human can inspect/edit
- ✅ Structured (markdown sections)
- ✅ Agent controls when to add info
- ✅ Persistent beyond subtasks
- ✅ No extra LLM calls
- ✅ Works with existing file-based workflow

### Cons
- ⚠️ Agent must remember to call `update_notes`
- ⚠️ Could get large over time (need size limit)
- ⚠️ All notes visible to all subtasks (not selective)
- ⚠️ Might duplicate with actual project files

### Variations
- **2a. Auto-cleanup**: System prompts agent to clean/organize notes every N subtasks
- **2b. Sections as tags**: Use tags for filtering (only show relevant sections)
- **2c. Read-only mode**: Notes locked during subtask, only editable at transitions
- **2d. Template sections**: Pre-create sections: "Findings", "Decisions", "TODO", "Resources"

---

## Proposal 3: Structured Artifact System

### Mechanism
System maintains typed artifacts that get selectively included in context:

**Artifact types**:
```python
- FINDING: "Discovered X in file Y"
- DECISION: "Chose to use approach Z because..."
- RESOURCE: "Important file: /path/to/file.py contains..."
- ERROR: "Hit issue: ..., workaround: ..."
- TODO: "Still need to: ..."
```

**During subtask**, agent calls:
```python
record_artifact(
    type="FINDING",
    content="Project uses FastAPI framework",
    relevance="architecture"
)
```

**On next subtask**, system analyzes subtask description and auto-includes relevant artifacts:
```
Subtask: "Add new API endpoint"

[Context includes relevant artifacts]
RELEVANT FINDINGS:
• Project uses FastAPI framework [from Subtask 1]
• Existing endpoints in routes/api.py [from Subtask 2]

RELEVANT RESOURCES:
• routes/api.py - API endpoint definitions [from Subtask 2]
```

### Implementation
```python
class Artifact:
    type: str  # FINDING, DECISION, RESOURCE, ERROR, TODO
    content: str
    relevance: str  # tag for filtering
    subtask_id: str
    timestamp: float

# Store in .agent_context/artifacts.json

# Smart inclusion based on relevance
def get_relevant_artifacts(subtask_desc: str, artifacts: list[Artifact]) -> list[Artifact]:
    # Simple keyword matching or embedding similarity
    keywords = extract_keywords(subtask_desc)
    return [a for a in artifacts if any(kw in a.relevance or kw in a.content for kw in keywords)]
```

### Pros
- ✅ Selective - only relevant info included
- ✅ Typed - clear categories
- ✅ Scalable - smart filtering prevents overload
- ✅ Inspectable - artifacts stored in JSON
- ✅ Flexible - agent chooses what to record

### Cons
- ⚠️ Agent must remember to record artifacts
- ⚠️ Relevance tagging requires agent judgment
- ⚠️ Filtering logic needs tuning (false negatives/positives)
- ⚠️ More complex to implement

### Variations
- **3a. Auto-extraction**: System analyzes agent responses and auto-creates artifacts
- **3b. Required artifacts**: Some subtask types require certain artifact types
- **3c. Artifact relationships**: Link artifacts (e.g., DECISION references FINDING)

---

## Proposal 4: Completion Summary + Smart Loading (Hybrid)

### Mechanism
Combines auto-summarization with smart retrieval:

**On completion**:
```
System: "What should future subtasks know about this work?"
Agent: [Provides 2-3 bullets]
System: [Stores as subtask.completion_summary]
```

**On new subtask start**:
```python
# System analyzes new subtask description
new_subtask = "Add authentication to API endpoints"

# Retrieves relevant past summaries using keyword/semantic matching
relevant = find_relevant_summaries(new_subtask, past_summaries)

# Includes ONLY relevant summaries in context
```

**Example**:
```
RELEVANT CONTEXT FROM PREVIOUS WORK:
• [Subtask 2] Found API endpoints defined in routes/api.py
• [Subtask 3] Decided to use JWT for auth (matches FastAPI patterns)
```

### Implementation
```python
# On completion
def mark_subtask_complete(success, reason):
    if success:
        # Get completion summary
        summary = get_completion_summary(current_subtask)
        current_subtask.summary = summary

    # Advance...

# On new subtask
def build_context(subtask):
    # Get relevant past summaries
    relevant_summaries = find_relevant(
        subtask.description,
        all_completed_subtask_summaries,
        max_results=3
    )

    if relevant_summaries:
        context.append({
            "role": "user",
            "content": f"RELEVANT CONTEXT:\n{format_summaries(relevant_summaries)}"
        })
```

### Pros
- ✅ Automatic - agent can't forget
- ✅ Selective - only relevant info
- ✅ Scales well - doesn't grow unbounded
- ✅ Natural - happens at boundaries
- ✅ Human-readable summaries

### Cons
- ⚠️ Extra LLM call per subtask
- ⚠️ Relevance matching needs tuning
- ⚠️ Might miss important context (false negatives)

### Variations
- **4a. Explicit + implicit**: Allow both auto-summaries AND manual artifact recording
- **4b. Confidence scores**: Show relevance score with each loaded summary
- **4c. Feedback loop**: Track which summaries were actually used (via read patterns)

---

## Proposal 5: Context Backpack (Carry-Forward)

### Mechanism
A special "backpack" object that persists across subtasks, but stays bounded.

**On subtask start**:
```
BACKPACK (max 500 tokens):
• Project uses FastAPI
• Main file: server.py
• Auth decided: JWT tokens
```

**During subtask**, agent can:
```python
add_to_backpack("Database uses PostgreSQL with SQLAlchemy ORM")
remove_from_backpack("Main file: server.py")  # No longer relevant
```

**System enforces size limit** (e.g., 500 tokens). If full, agent must remove items to add new ones.

### Implementation
```python
class Backpack:
    items: list[str]
    max_tokens: int = 500

    def add(self, item: str):
        if self.would_exceed(item):
            return {"error": "Backpack full. Remove items first."}
        self.items.append(item)

    def remove(self, index: int):
        del self.items[index]

    def clear(self):
        self.items = []

# In context
def build_context():
    if backpack.items:
        context.append({
            "role": "user",
            "content": f"BACKPACK ({len(backpack.items)} items):\n" +
                      "\n".join(f"• {item}" for item in backpack.items)
        })
```

### Pros
- ✅ Bounded size - can't grow forever
- ✅ Agent curates - decides what's important
- ✅ Always visible - in every subtask
- ✅ Simple mental model - like carrying notes

### Cons
- ⚠️ Agent must actively manage (add/remove)
- ⚠️ Fixed size might be too limiting
- ⚠️ Everything visible to all subtasks (not selective)
- ⚠️ Requires discipline to keep organized

### Variations
- **5a. Auto-compact**: System offers to summarize backpack when getting full
- **5b. Categories**: Separate backpacks for "facts", "decisions", "todos"
- **5c. Priority levels**: Important items less likely to be evicted

---

## Comparison Matrix

| Proposal | Automatic? | Selective? | Complexity | Extra LLM Calls | Human Control |
|----------|-----------|-----------|------------|----------------|---------------|
| 1. Auto-Summarization | ✅ Yes | ⚠️ No | Low | Yes (1/subtask) | Low |
| 2. Notes File | ⚠️ Manual | ⚠️ No | Low | No | High |
| 3. Structured Artifacts | ⚠️ Manual | ✅ Yes | High | No | High |
| 4. Hybrid (Summary+Smart) | ✅ Yes | ✅ Yes | Medium | Yes (1/subtask) | Low |
| 5. Backpack | ⚠️ Manual | ⚠️ No | Low | No | High |

---

## Recommended Approach

### **Proposal 4 (Hybrid) + Proposal 2 (Notes)**

**Rationale**: Combine automatic and manual mechanisms:

1. **Default (Automatic)**: Completion summaries + smart loading
   - Catches most cases
   - Zero agent effort
   - Works for discoveries, findings, decisions

2. **Override (Manual)**: Notes file for explicit sharing
   - For critical info agent wants to ensure is passed
   - For structured documentation
   - For TODO tracking

**Implementation**:
```python
# On mark_subtask_complete:
if success:
    # Auto-summarization
    summary = prompt_for_summary()  # "What should future subtasks know?"
    subtask.completion_summary = summary

# On new subtask start:
context = [
    system_prompt,
    goal_info,

    # 1. Shared notes (if exists)
    read_notes_file(),

    # 2. Relevant past summaries (auto-loaded)
    get_relevant_summaries(current_subtask.description),

    # 3. Current subtask
    current_subtask.description,

    # Empty message history (isolated!)
]
```

### Why This Works

1. **Foolproof**: Auto-summarization catches things agent doesn't explicitly think to share
2. **Selective**: Smart loading only includes relevant info
3. **Escape hatch**: Notes file for critical/structured info
4. **Bounded**: Relevance filtering + decay keeps it manageable
5. **Inspectable**: Both summaries and notes visible in state file
6. **Natural**: Minimal changes to agent workflow

### Refinements

**Summary Prompt**:
```
"This subtask is complete. Briefly summarize (2-3 points):
1. Key findings or discoveries
2. Important decisions made
3. Resources/files that matter for future work

Keep it concise - just the essentials future subtasks might need."
```

**Relevance Matching** (simple keyword approach):
```python
def get_relevant_summaries(subtask_desc, past_summaries, max=3):
    # Extract keywords
    keywords = extract_keywords(subtask_desc.lower())

    # Score each summary
    scored = []
    for summary in past_summaries:
        score = sum(1 for kw in keywords if kw in summary.text.lower())
        if score > 0:
            scored.append((score, summary))

    # Return top N
    return [s for _, s in sorted(scored, reverse=True)[:max]]
```

**Decay**: Only consider last 10 completed subtasks (prevent noise from early exploration).

---

## Implementation Phases

### Phase 1: Basic Auto-Summarization
- Add summary prompt on completion
- Store in subtask object
- Include ALL summaries in next subtask (no filtering yet)
- Validate it works, tune prompt

### Phase 2: Notes File Support
- Add `update_notes` tool
- Auto-load NOTES.md in context
- Let agents experiment with it

### Phase 3: Smart Loading
- Implement relevance matching
- Only load top 3 relevant summaries
- Add decay (ignore old subtasks)

### Phase 4: Refinement
- Tune prompts based on observation
- Adjust relevance algorithm
- Add structured artifact support if needed

---

## Open Questions

1. **How much context is too much?**
   - Need to monitor token usage
   - May need stricter limits than "top 3"

2. **What if relevance matching fails?**
   - Fallback: show last 2 summaries regardless
   - Or: let agent explicitly request past summaries

3. **Should notes file have schema?**
   - Start freeform, add structure if needed
   - Could template it: ## Findings, ## Decisions, ## TODO

4. **How to handle task-level vs subtask-level context?**
   - Task-level: Persistent across all subtasks in task
   - Subtask-level: Only for immediate next subtask
   - Proposal: Summaries are task-scoped

5. **Multi-agent system?**
   - Same approach should work for TaskExecutor
   - Orchestrator probably doesn't need it (uses append-until-full)

---

## Success Criteria

A good solution should:
- ✅ Work without agent needing to "remember" to share
- ✅ Not leak irrelevant information
- ✅ Be inspectable by humans
- ✅ Stay bounded (not grow forever)
- ✅ Feel natural to agent workflow
- ✅ Handle 90% of cases automatically
- ✅ Provide override for remaining 10%

**Proposal 4 + 2 meets all criteria.**

---

## Next Steps

1. **Review proposals** with human
2. **Choose approach** (recommend: Hybrid + Notes)
3. **Design detailed spec** for chosen approach
4. **Implement Phase 1** (basic auto-summary)
5. **Test and iterate** on prompt/approach
6. **Add Phase 2+3** (notes, smart loading)

---

**Author**: Claude (Sonnet 4.5)
**Date**: 2025-10-29
**Status**: Proposals for review - no implementation yet
