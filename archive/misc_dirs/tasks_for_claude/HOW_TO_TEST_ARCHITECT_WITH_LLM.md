# How to Test Architect Integration with Actual LLM

## Quick Start

Once Ollama is running, test the complete orchestrator → architect → task executor workflow:

### 1. Start Ollama
```bash
ollama serve
```

### 2. Run Orchestrator with Complex Project

```bash
python orchestrator_main.py --once "web app that renders procedural art with gpu acceleration and user interactive live settings"
```

## What Should Happen

### Step 1: Orchestrator Complexity Assessment
The orchestrator should recognize this as a complex project:
- Multi-component system (UI + GPU renderer + settings)
- Technology stack decisions needed
- Performance concerns (GPU acceleration)
- Complex integration requirements

**Expected**: Orchestrator decides to consult architect

### Step 2: Architect Consultation
The orchestrator calls `consult_architect` with:
```python
{
    "project_description": "web app that renders procedural art with gpu acceleration and user interactive live settings",
    "requirements": "...",
    "constraints": "..."
}
```

**Expected**: Architect agent starts consultation session

### Step 3: Architect Creates Artifacts
The architect should:
1. **Analyze requirements** - understand what's needed
2. **Ask clarifying questions** (optional) - if anything is unclear
3. **Design architecture** - choose technologies, identify modules
4. **Create documentation**:
   - `architecture/system-overview.md` - high-level design
   - `architecture/modules/gpu-renderer.md` - GPU rendering spec
   - `architecture/modules/web-ui.md` - UI component spec
   - `architecture/modules/settings-manager.md` - real-time settings spec
5. **Create task breakdown**:
   - `architecture/task-breakdown.json` - structured task list

**Expected output in workspace**:
```
.agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/
├── architecture/
│   ├── system-overview.md
│   ├── modules/
│   │   ├── gpu-renderer.md
│   │   ├── web-ui.md
│   │   └── settings-manager.md
│   └── task-breakdown.json
└── (implementation files will be added by task executor)
```

### Step 4: Orchestrator Reads Task Breakdown
The orchestrator should read `architecture/task-breakdown.json`:
```json
{
    "tasks": [
        {
            "id": "T1",
            "description": "Implement GPU renderer module",
            "module": "gpu-renderer",
            "priority": 1,
            "dependencies": []
        },
        {
            "id": "T2",
            "description": "Implement web UI",
            "module": "web-ui",
            "priority": 2,
            "dependencies": ["T1"]
        },
        {
            "id": "T3",
            "description": "Implement settings manager",
            "module": "settings-manager",
            "priority": 2,
            "dependencies": ["T1"]
        }
    ]
}
```

### Step 5: Orchestrator Delegates to Task Executor
For each task in the breakdown, orchestrator should call:
```python
delegate_to_executor(
    task_description="Implement GPU renderer module per architecture/modules/gpu-renderer.md",
    workspace_mode="existing",
    workspace_path=".agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel"
)
```

**Expected**: Task executor implements each module according to architecture specs

### Step 6: Final Result
All work should be in a single workspace:
```
.agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/
├── architecture/                    # From Architect
│   ├── system-overview.md
│   ├── modules/*.md
│   └── task-breakdown.json
├── src/                            # From Task Executor
│   ├── renderer/
│   │   └── gpuRenderer.js          # WebGL-based procedural art
│   ├── ui/
│   │   └── App.jsx                 # React UI with controls
│   └── settings/
│       └── settingsManager.js      # Real-time updates
├── package.json
├── index.html
└── tests/
```

## Expected Console Output

```
============================================================
JETBOX ORCHESTRATOR
============================================================

You: web app that renders procedural art with gpu acceleration and user interactive live settings

Orchestrator: This is a complex multi-component project. Let me consult the architect...

[Calling consult_architect]

============================================================
ARCHITECT CONSULTATION
============================================================
Project: web app that renders procedural art with gpu acceleration...
============================================================

[architect] Round 1/10
[architect] Creating architecture/system-overview.md

[architect] Round 2/10
[architect] Creating architecture/modules/gpu-renderer.md

[architect] Round 3/10
[architect] Creating architecture/modules/web-ui.md

[architect] Round 4/10
[architect] Creating architecture/modules/settings-manager.md

[architect] Round 5/10
[architect] Creating architecture/task-breakdown.json

[Consultation complete]

Orchestrator: Architecture design complete! Now implementing...

[Delegating to task executor - Task 1]
[Task executor working...]
[Task executor complete]

[Delegating to task executor - Task 2]
[Task executor working...]
[Task executor complete]

[Delegating to task executor - Task 3]
[Task executor working...]
[Task executor complete]

Orchestrator: Project complete! Your procedural art web app is in:
  .agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/
```

## Verification Steps

After the orchestrator completes:

### 1. Check Architecture Documentation
```bash
cd .agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel
cat architecture/system-overview.md
cat architecture/modules/gpu-renderer.md
cat architecture/task-breakdown.json
```

**Expected**: Detailed architecture documentation with:
- Component descriptions
- Technology choices (WebGL, React, etc.)
- Interface specifications
- Implementation guidance

### 2. Check Implementation
```bash
ls -R src/
cat package.json
```

**Expected**: Working implementation matching architecture specs

### 3. Test the App
```bash
npm install
npm run dev
# Open browser to http://localhost:3000 or similar
```

**Expected**: Working procedural art web app with:
- GPU-accelerated rendering (WebGL/WebGPU)
- Interactive controls (sliders, buttons)
- Real-time parameter updates
- Procedural art patterns (noise, fractals, etc.)

## Troubleshooting

### Orchestrator Doesn't Consult Architect
**Symptom**: Orchestrator delegates directly to task executor without consulting architect

**Causes**:
- LLM didn't recognize complexity
- System prompt unclear

**Solution**: Request is truly complex enough, adjust orchestrator prompt or try more explicit phrasing:
```bash
python orchestrator_main.py --once "design and build a complex web application with gpu-accelerated procedural art rendering, real-time interactive controls, and websocket-based live parameter updates"
```

### Architect Doesn't Create Artifacts
**Symptom**: Architect runs but creates no files

**Causes**:
- LLM didn't call tools
- Temperature too high (hallucinating instead of acting)

**Solution**: Check architect system prompt, verify tools are available

### Task Executor Can't Find Architecture Docs
**Symptom**: Task executor fails with "architecture/modules/X.md not found"

**Causes**:
- Workspace path not passed correctly
- Architect didn't create files

**Solution**: Verify workspace path in delegation calls, check architect actually wrote files

## Alternative: Interactive Mode

For more control, run without `--once`:

```bash
python orchestrator_main.py
```

Then at the prompt:
```
You: I want to build a complex web app for gpu-accelerated procedural art with real-time interactive settings. This is a multi-component system.

Orchestrator: [assesses complexity]

You: [follow along, provide clarifications if architect asks questions]
```

## Test Cases

### Complex Projects (Should Trigger Architect)
- ✅ "web app that renders procedural art with gpu acceleration and user interactive live settings"
- ✅ "real-time analytics platform with streaming data ingestion and dashboards"
- ✅ "microservices architecture with auth, API gateway, and event-driven messaging"
- ✅ "e-commerce platform with inventory management, payment processing, and order tracking"

### Simple Projects (Should Skip Architect)
- ❌ "simple calculator script in Python"
- ❌ "todo list CRUD app with SQLite"
- ❌ "fix the bug in auth.py where passwords aren't hashed"
- ❌ "add a /health endpoint to the API"

## Success Criteria

✅ **Full workflow executes**: User → Orchestrator → Architect → Orchestrator → TaskExecutor → User

✅ **Architecture artifacts created**: Docs, module specs, task breakdown in workspace/architecture/

✅ **Implementation follows architecture**: TaskExecutor implements modules per specs

✅ **Single shared workspace**: All artifacts and code in one directory

✅ **Working application**: Final result runs and meets requirements

## Next Steps After Success

Once the basic workflow works, you can enhance:

1. **Multiple iterations**: User can request changes, orchestrator re-consults architect
2. **Dependency-aware delegation**: Orchestrator respects task dependencies (T2 waits for T1)
3. **Progress reporting**: Orchestrator shows which tasks are complete
4. **Architecture refinement**: Architect can update specs based on implementation feedback

---

## Quick Command Reference

```bash
# Start Ollama
ollama serve

# Test complex project (architect should be consulted)
python orchestrator_main.py --once "web app that renders procedural art with gpu acceleration and user interactive live settings"

# Interactive mode
python orchestrator_main.py

# Check what was created
ls -R .agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/

# View architecture
cat .agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/architecture/system-overview.md

# View task breakdown
cat .agent_workspace/web-app-that-renders-procedural-art-with-gpu-accel/architecture/task-breakdown.json

# Run simulation (no LLM needed)
python tests/test_procedural_art_workflow.py
```
