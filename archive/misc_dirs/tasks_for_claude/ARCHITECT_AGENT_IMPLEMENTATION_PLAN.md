# Architect Agent Implementation Plan

## Executive Summary

Create a **Software Architect agent** that sits between the Orchestrator and TaskExecutor, providing architecture design, technology recommendations, and decomposition guidance for complex projects. The Architect produces structured artifacts (architecture docs, module specs, task breakdowns) that the Orchestrator can use to coordinate multiple TaskExecutor delegations.

---

## Current Architecture

```
User
  ↓
Orchestrator (conversational, delegates work)
  ↓
TaskExecutor (executes coding tasks)
```

**Limitations:**
- Orchestrator delegates directly to TaskExecutor without architectural planning
- No structured approach for complex multi-component systems
- TaskExecutor receives raw goals, must figure out architecture on the fly
- No separation between "what to build" (architecture) and "how to build it" (implementation)

---

## Proposed Architecture

```
User
  ↓
Orchestrator (conversational, routes to appropriate agent)
  ├─→ Architect (for complex projects: design architecture)
  │     ↓ (produces: architecture doc, module specs, task list)
  │     └─→ Return to Orchestrator
  └─→ TaskExecutor (execute specific tasks)
        └─→ Can be called multiple times for different modules
```

**Key Flow:**
1. User asks Orchestrator for complex system
2. Orchestrator recognizes complexity, delegates to Architect
3. Architect produces architecture artifacts + task breakdown
4. Orchestrator delegates individual tasks to TaskExecutor(s)
5. Results combined and presented to user

---

## Design Decisions

### 1. **Agent Type: Consultant (Non-Terminal)**

**Architect Agent**:
- **Does NOT execute code** (unlike TaskExecutor)
- **Produces artifacts**: Architecture docs, module specs, task lists
- **Returns control to Orchestrator** (consultant model)
- **Stateless**: Each invocation is independent

**Rationale:**
- Separation of concerns: architecture design ≠ implementation
- Orchestrator remains in control of overall workflow
- Can consult Architect multiple times for different aspects
- Artifacts persist in workspace for reference

### 2. **Context Strategy: Custom "ArchitectStrategy"**

**Requirements:**
- Keep full conversation (like Orchestrator's AppendUntilFullStrategy)
- Support iterative refinement (user asks clarifying questions)
- Manage large artifacts (architecture docs can be verbose)
- Summary/archive capability when context grows too large

**Implementation:**
```python
class ArchitectStrategy(ContextStrategy):
    """
    Strategy for architecture design conversations.

    Features:
    - Append-until-full with intelligent compaction
    - Preserves architecture decisions and artifacts
    - Can archive old design iterations
    - Does NOT use jetbox notes (artifacts are the output)
    """

    def should_use_jetbox_notes(self) -> bool:
        return False  # Architect produces artifacts, not summaries
```

### 3. **Architect-Specific Tools**

**Core Tools:**
1. **`write_architecture_doc`** - Write architecture overview to workspace
2. **`write_module_spec`** - Write detailed module specification
3. **`write_task_list`** - Generate structured task breakdown
4. **`archive_design_iteration`** - Archive old design iteration when pivoting
5. **`read_codebase_structure`** - Inspect existing codebase structure (for refactors)
6. **`read_existing_docs`** - Read existing architecture docs (for iterations)

**Extended Tools (future):**
7. **`query_tech_stack_compatibility`** - Check technology compatibility
8. **`estimate_complexity`** - Estimate implementation complexity
9. **`generate_api_contracts`** - Generate OpenAPI/interface specs

### 4. **Artifact Storage**

**Workspace Structure:**
```
.agent_workspace/{project-slug}/
├── architecture/
│   ├── overview.md              # High-level architecture
│   ├── decisions.md             # Architecture Decision Records (ADRs)
│   ├── modules/
│   │   ├── module-auth.md       # Module specifications
│   │   ├── module-api.md
│   │   └── module-db.md
│   └── archive/                 # Old design iterations
│       └── v1-initial-design.md
├── tasks/
│   └── breakdown.json           # Structured task list for Orchestrator
└── jetboxnotes.md               # (not used by Architect)
```

---

## Implementation Plan

### **Phase 1: Core Agent & Integration**

#### 1.1 Create ArchitectAgent Class

**File**: `architect_agent.py`

```python
class ArchitectAgent(BaseAgent):
    """
    Agent specialized for software architecture design.

    Responsibilities:
    - Gather requirements and constraints
    - Propose architecture patterns and trade-offs
    - Design module structure and interfaces
    - Produce architecture artifacts
    - Generate structured task breakdown

    Context strategy: ArchitectStrategy (append-until-full with archiving)
    Tools: Architecture artifact creation, codebase inspection
    """

    def __init__(self, workspace: Path):
        super().__init__(
            name="architect",
            role="Software architecture consultant",
            workspace=workspace,
            config=config,
        )
        self.context_strategy = ArchitectStrategy()

    def get_tools(self) -> list[dict[str, Any]]:
        """Return architect-specific tools."""
        return [
            # Architecture artifact tools
            self._tool_write_architecture_doc(),
            self._tool_write_module_spec(),
            self._tool_write_task_list(),
            self._tool_archive_design_iteration(),

            # Inspection tools (read-only)
            self._tool_read_codebase_structure(),
            self._tool_read_existing_docs(),

            # Base tools (read_file, list_dir)
            *get_base_inspection_tools(),
        ]
```

#### 1.2 Create ArchitectStrategy

**File**: `context_strategies.py` (extend)

```python
class ArchitectStrategy(ContextStrategy):
    """
    Context strategy for architecture design conversations.

    Characteristics:
    - Append all messages until near token limit
    - Compact old iterations when context grows
    - Preserve key decisions and current architecture state
    - No jetbox notes (artifacts are persistent output)
    """

    def __init__(self, max_tokens: int = 32000, archive_threshold: int = 20):
        self.max_tokens = max_tokens
        self.archive_threshold = archive_threshold  # messages before archiving

    def should_use_jetbox_notes(self) -> bool:
        return False  # Architect produces artifacts, not summaries

    def build_context(self, ...):
        # Similar to AppendUntilFullStrategy but:
        # - Higher token limit (architecture docs are verbose)
        # - Archive capability for design iterations
        # - Include references to artifacts in workspace
        ...
```

#### 1.3 Update agents.yaml

```yaml
agents:
  orchestrator:
    class: OrchestratorAgent
    description: "Manages user conversation and delegates work"
    can_delegate_to:
      - architect      # NEW: Can consult architect
      - task_executor

  architect:            # NEW AGENT
    class: ArchitectAgent
    description: "Software architecture consultant for complex projects"
    can_delegate_to: []  # Terminal consultant (doesn't delegate)

  task_executor:
    class: TaskExecutorAgent
    description: "Executes coding tasks"
    can_delegate_to: []
```

#### 1.4 Update Orchestrator Tools

**Add tool**: `consult_architect`

```python
{
    "name": "consult_architect",
    "description": "Consult the Architect agent for complex project design. Use when user requests multi-component systems, refactoring large codebases, or needs technology recommendations.",
    "parameters": {
        "project_description": "string",  # What to build
        "requirements": "string",         # Functional + non-functional
        "constraints": "string",          # Team size, tech stack, timeline
        "existing_context": "string"      # Existing code/docs (optional)
    }
}
```

**Tool handler**: Similar to `delegate_to_executor`, but:
- Doesn't run task to completion
- Returns architecture artifacts (file paths)
- Orchestrator can ask follow-up questions
- Multiple rounds of refinement supported

---

### **Phase 2: Architect Tools Implementation**

#### 2.1 Architecture Artifact Tools

**File**: `architect_tools.py`

```python
def write_architecture_doc(
    title: str,
    content: str,
    workspace_manager: WorkspaceManager
) -> dict:
    """
    Write high-level architecture document.

    Args:
        title: Document title (e.g., "System Overview", "Data Flow Architecture")
        content: Markdown content with diagrams, component descriptions
        workspace_manager: Workspace to write to

    Returns:
        {"status": "success", "file_path": "architecture/overview.md"}
    """
    arch_dir = workspace_manager.workspace_dir / "architecture"
    arch_dir.mkdir(exist_ok=True)

    file_path = arch_dir / f"{slugify(title)}.md"
    file_path.write_text(content, encoding="utf-8")

    return {
        "status": "success",
        "file_path": str(file_path.relative_to(workspace_manager.workspace_dir)),
        "message": f"Architecture doc written: {title}"
    }


def write_module_spec(
    module_name: str,
    responsibility: str,
    interfaces: dict,
    dependencies: list[str],
    technologies: dict,
    workspace_manager: WorkspaceManager
) -> dict:
    """
    Write detailed module specification.

    Args:
        module_name: Module identifier (e.g., "auth-service")
        responsibility: What this module does
        interfaces: {"inputs": [...], "outputs": [...], "apis": [...]}
        dependencies: List of other modules this depends on
        technologies: {"language": "Python", "framework": "FastAPI", ...}
        workspace_manager: Workspace

    Returns:
        {"status": "success", "file_path": "architecture/modules/auth-service.md"}
    """
    # Generate structured module spec markdown
    content = f"""# Module: {module_name}

## Responsibility
{responsibility}

## Interfaces
### Inputs
{yaml.dump(interfaces.get('inputs', []))}

### Outputs
{yaml.dump(interfaces.get('outputs', []))}

### APIs
{yaml.dump(interfaces.get('apis', []))}

## Dependencies
{chr(10).join(f'- {dep}' for dep in dependencies)}

## Technologies
{yaml.dump(technologies)}
"""

    modules_dir = workspace_manager.workspace_dir / "architecture" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    file_path = modules_dir / f"{slugify(module_name)}.md"
    file_path.write_text(content, encoding="utf-8")

    return {
        "status": "success",
        "file_path": str(file_path.relative_to(workspace_manager.workspace_dir)),
    }


def write_task_list(
    tasks: list[dict],
    workspace_manager: WorkspaceManager
) -> dict:
    """
    Write structured task breakdown for Orchestrator.

    Args:
        tasks: [{"id": "T1", "description": "...", "module": "...", "priority": 1}, ...]
        workspace_manager: Workspace

    Returns:
        {"status": "success", "file_path": "tasks/breakdown.json", "task_count": 5}
    """
    tasks_dir = workspace_manager.workspace_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    file_path = tasks_dir / "breakdown.json"
    with open(file_path, "w") as f:
        json.dump({"tasks": tasks, "generated_at": datetime.now().isoformat()}, f, indent=2)

    return {
        "status": "success",
        "file_path": str(file_path.relative_to(workspace_manager.workspace_dir)),
        "task_count": len(tasks),
        "message": f"Task breakdown written: {len(tasks)} tasks"
    }
```

#### 2.2 Inspection Tools (Read-Only)

```python
def read_codebase_structure(
    path: str,
    workspace_manager: WorkspaceManager
) -> dict:
    """
    Inspect existing codebase structure.

    Returns:
        {
            "directories": [...],
            "files": [...],
            "languages": {...},  # File count by extension
            "structure_summary": "..."
        }
    """
    # Walk directory tree, count files by type, identify key dirs
    ...


def read_existing_docs(
    doc_path: str,
    workspace_manager: WorkspaceManager
) -> dict:
    """
    Read existing architecture docs for iteration.

    Returns:
        {"content": "...", "file_path": "..."}
    """
    ...
```

---

### **Phase 3: Orchestrator Integration**

#### 3.1 Update Orchestrator System Prompt

**Add section** on when to consult architect:

```
WHEN TO CONSULT ARCHITECT:
Use consult_architect when the user requests:
- Multi-component systems (e.g., "build a microservices platform")
- Complex data flows (e.g., "real-time analytics pipeline")
- Technology stack decisions (e.g., "should I use SQL or NoSQL?")
- Refactoring large codebases
- Performance/scaling concerns
- Multi-phase projects requiring roadmap

DO NOT consult architect for:
- Simple single-file scripts
- Small feature additions to existing code
- Bug fixes
- Straightforward CRUD applications

WORKFLOW WITH ARCHITECT:
1. Identify complexity → consult_architect(description, requirements, constraints)
2. Review architecture artifacts returned
3. Ask follow-up questions if needed (architect remains available)
4. Once architecture finalized, delegate modules to task_executor
5. Report architecture decisions and file locations to user
```

#### 3.2 Orchestrator Tool Handler

**File**: `orchestrator_main.py` (extend `execute_orchestrator_tool`)

```python
elif tool_name == "consult_architect":
    # Similar to delegate_to_executor but for consultation
    project_description = args.get("project_description", "")
    requirements = args.get("requirements", "")
    constraints = args.get("constraints", "")
    existing_context = args.get("existing_context", "")

    # Set up architect consultation
    result = registry.consult_agent(
        from_agent="orchestrator",
        to_agent="architect",
        project_description=project_description,
        requirements=requirements,
        constraints=constraints,
        existing_context=existing_context,
    )

    # Architect runs interactively until it produces artifacts
    # Returns paths to architecture docs, module specs, task list

    if result["success"]:
        # Read artifacts and return to orchestrator
        artifacts = {
            "architecture_docs": result.get("architecture_docs", []),
            "module_specs": result.get("module_specs", []),
            "task_list": result.get("task_list"),
        }

        message = f"Architecture consultation complete.\n"
        message += f"Documents: {', '.join(artifacts['architecture_docs'])}\n"
        message += f"Modules: {len(artifacts['module_specs'])}\n"
        message += f"Tasks: {artifacts['task_list']}"

        return {
            "success": True,
            "artifacts": artifacts,
            "message": message,
        }
```

---

### **Phase 4: Example Workflows**

#### Workflow 1: Simple Project (No Architect)

```
User: "Create a todo list app with FastAPI"
Orchestrator: [recognizes simple project]
  → delegate_to_executor("Create todo list app with FastAPI")
TaskExecutor: [builds the app]
Orchestrator: "Task complete. Files in .agent_workspace/todo-list-app/"
```

#### Workflow 2: Complex Project (With Architect)

```
User: "Build a real-time analytics platform with multi-tenant isolation"
Orchestrator: [recognizes complexity]
  → consult_architect(
      project_description="real-time analytics platform",
      requirements="1M events/sec, multi-tenant isolation, schema evolution",
      constraints="AWS + Kubernetes, 4-person team, Go/Python"
    )

Architect: [asks clarifications]
  - "What's the latency target for insights?"
  - "What defines a tenant?"
  - "Regulatory constraints?"

User: [answers via Orchestrator]

Architect: [produces artifacts]
  ✓ architecture/overview.md (stream processing architecture)
  ✓ architecture/modules/ingestion.md
  ✓ architecture/modules/processing.md
  ✓ architecture/modules/storage.md
  ✓ architecture/modules/api.md
  ✓ tasks/breakdown.json (15 tasks)

Orchestrator: [reads task breakdown]
  → delegate_to_executor("Implement ingestion module per spec")
  → delegate_to_executor("Implement processing module per spec")
  → delegate_to_executor("Implement storage module per spec")
  → delegate_to_executor("Implement API module per spec")

Orchestrator: "Architecture complete. Modules implemented. See .agent_workspace/analytics-platform/"
```

#### Workflow 3: Iterative Architecture

```
User: "I need to refactor my monolith to microservices"
Orchestrator:
  → consult_architect(
      project_description="Refactor monolith to microservices",
      existing_context="Existing codebase at ./myapp/"
    )

Architect: [inspects codebase]
  → read_codebase_structure("./myapp/")

Architect: "I recommend a strangler fig pattern with 4 bounded contexts..."
User: "Show me the first phase"

Architect: [refines]
  ✓ architecture/phased-migration.md (3 phases)
  ✓ architecture/modules/phase1-auth-service.md
  ✓ tasks/phase1-tasks.json

Orchestrator: "Phase 1 plan ready. Proceed with implementation?"
User: "Yes"

Orchestrator:
  → delegate_to_executor("Extract auth service per phase1 spec")
```

---

### **Phase 5: Advanced Features (Future)**

#### 5.1 Multi-Agent Collaboration

**Scenario**: Architect and TaskExecutor collaborate

```
Orchestrator → Architect: "Design e-commerce platform"
Architect: [produces architecture]

Orchestrator → TaskExecutor: "Implement payment module"
TaskExecutor: [encounters issue with payment gateway integration]
  → TaskExecutor sends message to Orchestrator
  → Orchestrator re-consults Architect

Architect: "Recommend using Stripe SDK with retry queue..."
Orchestrator → TaskExecutor: [provides architecture guidance]
TaskExecutor: [continues with new approach]
```

#### 5.2 Architecture Validation Agent

**New agent**: `ArchitectValidator`
- Reviews TaskExecutor output against architecture spec
- Ensures implementation matches design
- Reports deviations to Orchestrator

#### 5.3 Architect Learning from Implementation

```
TaskExecutor: [completes module, encounters performance issue]
Orchestrator: [reports back to Architect]
Architect: [updates architecture doc with lessons learned]
  → write_architecture_doc("Performance Lessons", "...")
```

---

## Implementation Checklist

### **Milestone 1: Basic Architect Agent (MVP)**
- [ ] Create `ArchitectAgent` class
- [ ] Create `ArchitectStrategy` context strategy
- [ ] Implement `write_architecture_doc` tool
- [ ] Implement `write_module_spec` tool
- [ ] Implement `write_task_list` tool
- [ ] Update `agents.yaml` with architect agent
- [ ] Update `agent_registry.py` to support architect
- [ ] Create basic system prompt for architect
- [ ] Test: Simple consultation produces artifacts

### **Milestone 2: Orchestrator Integration**
- [ ] Add `consult_architect` tool to Orchestrator
- [ ] Update Orchestrator system prompt with architect workflow
- [ ] Implement `execute_orchestrator_tool` handler for architect
- [ ] Test: Orchestrator can delegate to architect
- [ ] Test: Orchestrator can read architect artifacts
- [ ] Test: End-to-end workflow with user

### **Milestone 3: Advanced Tools**
- [ ] Implement `read_codebase_structure` tool
- [ ] Implement `read_existing_docs` tool
- [ ] Implement `archive_design_iteration` tool
- [ ] Test: Architect can inspect existing code
- [ ] Test: Architect can iterate on previous designs

### **Milestone 4: Multi-Round Consultation**
- [ ] Support multiple rounds of architect consultation
- [ ] Architect context persistence between rounds
- [ ] Test: User asks follow-up questions
- [ ] Test: Architect refines based on feedback

### **Milestone 5: Task Breakdown Integration**
- [ ] Structured task list format (JSON schema)
- [ ] Orchestrator reads task list and delegates sequentially
- [ ] Progress tracking across multiple delegations
- [ ] Test: Full complex project workflow

---

## Technical Specifications

### Artifact Formats

#### Architecture Document (`architecture/overview.md`)

```markdown
# System Architecture: [Project Name]

## Overview
[High-level description]

## Architecture Pattern
[Microservices / Monolith / Event-Driven / Layered / ...]

## Components
### Component 1: [Name]
- **Responsibility**: ...
- **Technology**: ...
- **Dependencies**: ...

### Component 2: [Name]
...

## Data Flow
[Description with ASCII diagram]

## Non-Functional Requirements
### Scalability
...
### Security
...
### Maintainability
...

## Trade-offs
### Decision: [Choice Made]
- **Alternatives Considered**: ...
- **Rationale**: ...
- **Trade-offs**: ...

## Risks & Mitigation
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ... | ... | ... | ... |

## Roadmap
### Phase 1: MVP
- ...
### Phase 2: Scale
- ...
```

#### Module Specification (`architecture/modules/{module}.md`)

```markdown
# Module: [Module Name]

## Responsibility
[What this module does]

## Interfaces
### Inputs
- Input 1: type, source, format
- Input 2: ...

### Outputs
- Output 1: type, destination, format
- Output 2: ...

### APIs
- `POST /api/v1/resource` - Description
- `GET /api/v1/resource/{id}` - Description

## Dependencies
### Internal Dependencies
- Module A: for authentication
- Module B: for data access

### External Dependencies
- PostgreSQL: for data persistence
- Redis: for caching

## Technologies
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Database**: PostgreSQL 15
- **Testing**: pytest

## Implementation Notes
[Specific guidance for TaskExecutor]
```

#### Task Breakdown (`tasks/breakdown.json`)

```json
{
  "generated_at": "2025-10-31T12:00:00Z",
  "total_tasks": 15,
  "tasks": [
    {
      "id": "T1",
      "description": "Implement authentication module",
      "module": "auth-service",
      "priority": 1,
      "dependencies": [],
      "estimated_complexity": "medium",
      "files_to_create": ["auth/service.py", "auth/models.py"],
      "tests_required": true
    },
    {
      "id": "T2",
      "description": "Implement API gateway",
      "module": "api-gateway",
      "priority": 2,
      "dependencies": ["T1"],
      "estimated_complexity": "high",
      "files_to_create": ["gateway/app.py", "gateway/routes.py"],
      "tests_required": true
    }
  ]
}
```

---

## Success Criteria

### Functional Requirements
✅ Architect agent can be consulted by Orchestrator
✅ Architect produces structured architecture artifacts
✅ Artifacts are persisted in workspace
✅ Orchestrator can read and act on architect artifacts
✅ Multiple rounds of consultation supported
✅ Architect can inspect existing codebases

### Non-Functional Requirements
✅ Context management handles large architecture docs
✅ Artifacts are human-readable (Markdown)
✅ Task breakdowns are machine-readable (JSON)
✅ Response time < 30s for initial consultation
✅ Architect doesn't execute code (read-only + artifact creation)

### User Experience
✅ Clear workflow: complex project → consult architect → delegate tasks
✅ Architecture artifacts are useful reference docs
✅ Task breakdowns guide implementation
✅ User can iterate on architecture before implementation

---

## Summary

This plan creates a **Software Architect agent** that:

1. **Separates concerns**: Architecture design vs. implementation
2. **Produces artifacts**: Reusable docs, specs, task lists
3. **Integrates with Orchestrator**: Consultant model, not executor
4. **Enables complex projects**: Multi-component systems, refactoring
5. **Supports iteration**: Multiple rounds, refinement, feedback
6. **Scales gracefully**: Simple projects skip architect, complex projects benefit

**Key Innovation**: The Architect agent doesn't execute code - it produces *plans* that the Orchestrator uses to coordinate multiple TaskExecutor delegations. This mirrors real-world software development: architect designs, developers implement.

**Next Steps**:
1. Review and approve this plan
2. Implement Milestone 1 (Basic Architect Agent MVP)
3. Test with simple consultation workflow
4. Iterate based on feedback
