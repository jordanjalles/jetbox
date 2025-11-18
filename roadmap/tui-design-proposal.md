# Jetbox TUI Design Proposal

**Date**: 2025-11-18
**Status**: Design Exploration
**Purpose**: Comprehensive analysis of TUI options for enhanced developer experience

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [User Research](#user-research)
3. [Technical Approaches](#technical-approaches)
4. [UX Design](#ux-design)
5. [Visual Styles](#visual-styles)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Recommendation](#recommendation)

---

## Executive Summary

This proposal explores TUI (Terminal User Interface) options for Jetbox, analyzing:
- **5 technical approaches** (Textual, Rich, hybrid, web-based, tmux)
- **4 user personas** with distinct needs
- **5 visual style directions** (minimalist to IDE-like)
- **3 color scheme families** (classic, modern, customizable)

**TL;DR Recommendation**:
- **Phase 1**: Rich-based minimalist output (1 week, low risk)
- **Phase 2**: Textual-based interactive dashboard (2-3 weeks, high value)
- **Phase 3**: Advanced features (multi-agent, charts, exports)

This staged approach balances quick wins with long-term vision while minimizing risk.

---

## User Research

### Persona 1: Solo Developer (Primary User)

**Profile**:
- Runs single agent tasks (80% of usage)
- Wants quick feedback on progress
- Needs to know when to intervene
- Terminal is secondary window (editor is primary)

**Pain Points**:
- Current output is verbose, hard to scan
- Can't tell if agent is stuck or progressing
- No easy way to pause/inspect state
- Logs scroll off screen

**Needs**:
- Compact status summary (goal, round, time)
- Visual progress indicator
- Quick access to pause/resume
- Ability to scroll back through logs

**Use Cases**:
1. "Create a REST API" → check status every few minutes
2. Agent stuck in loop → pause, inspect context, abort
3. Task completed → see summary, verify files created

---

### Persona 2: Power User / Debugger

**Profile**:
- Developing/debugging Jetbox itself
- Needs deep visibility into agent internals
- Wants to trace decision-making
- Comfortable with complexity

**Pain Points**:
- Context inspection requires reading JSON files
- Can't step through rounds interactively
- Hard to correlate LLM response with tool calls
- No visibility into behavior lifecycle events

**Needs**:
- Full context dump viewer (with syntax highlighting)
- Step-by-step execution mode
- Behavior event logs
- Token usage breakdown by section
- Export capabilities (logs, context, stats)

**Use Cases**:
1. Agent fails unexpectedly → inspect exact prompt sent
2. Behavior bug suspected → trace on_before_llm_call events
3. Performance analysis → view token usage per round
4. Regression testing → export context, replay with different model

---

### Persona 3: Researcher Running Evaluations

**Profile**:
- Runs 10-50 agent tasks overnight
- Needs high-level progress tracking
- Wants summary statistics
- Monitors resource usage (GPU, costs)

**Pain Points**:
- No overview of multiple running agents
- Can't see aggregate stats (success rate, avg time)
- Hard to identify which tasks failed
- No alerts on critical failures

**Needs**:
- Multi-agent dashboard view
- Progress: X/N completed, Y in progress, Z pending
- Quick filtering: show only failures
- Resource monitoring (VRAM, tokens, cost)
- Notifications (desktop/webhook) on completion

**Use Cases**:
1. Start 50-task eval → see dashboard with queue
2. Morning: check overnight results → 45/50 succeeded
3. Investigate failures → click failed task, see logs
4. Generate report → export stats to JSON/CSV

---

### Persona 4: Team Lead (Future)

**Profile**:
- Manages multiple developers using Jetbox
- Deploys agents to production (cloud)
- Needs centralized monitoring
- Concerned about costs and security

**Pain Points**:
- No visibility into team's agent usage
- Can't track aggregate costs
- No audit trail for security events
- Hard to share agent status with stakeholders

**Needs**:
- Web-based dashboard (accessible remotely)
- Multi-user view (who's running what)
- Cost tracking per user/project
- Security event log (Rule of Two violations)
- Shareable links to agent runs

**Use Cases**:
1. Monthly review: view team's total token usage
2. Security audit: check if any agents accessed sensitive files
3. Stakeholder demo: share live link to agent building feature
4. Cost optimization: identify most expensive agent runs

---

## Technical Approaches

### Option 1: Textual (Modern Python TUI Framework)

**Description**: Reactive, component-based TUI framework (like React for terminals)

**Technology**:
- Framework: [Textual](https://textual.textualize.io/)
- Language: Python 3.11+
- Dependencies: Rich (rendering), asyncio (reactivity)
- Terminal: Supports most modern terminals (iTerm, Windows Terminal, etc.)

**Architecture**:
```python
class JetboxDashboard(App):
    def compose(self):
        yield Header()
        yield StatusPanel()
        yield Container(
            LogViewer(),
            FileTree(),
            ContextPreview(),
        )
        yield Footer()

    def on_mount(self):
        self.watch_agent(agent_id)

    def on_key(self, event: events.Key):
        if event.key == "p":
            self.agent.pause()
```

**Pros**:
- ✅ Modern, component-based architecture
- ✅ Built-in widgets (buttons, tables, trees, progress bars)
- ✅ Reactive data binding (auto-updates when state changes)
- ✅ Great documentation and examples
- ✅ Active development, growing community
- ✅ Supports mouse input
- ✅ CSS-like styling system

**Cons**:
- ❌ Learning curve (new paradigm for Python devs)
- ❌ Heavier dependency (Rich + Textual)
- ❌ Requires Python 3.11+ (may limit users)
- ❌ Performance overhead vs raw curses
- ❌ Still maturing (API may change)

**Best For**:
- Rich, interactive dashboards
- Multi-panel layouts with keyboard navigation
- Long-running agents with live updates
- Users who value polished UX

**Effort**: Medium-High (2-3 weeks for full dashboard)
**Risk**: Low (well-documented, active community)
**Maintenance**: Low (framework handles complexity)

---

### Option 2: Rich (Enhanced Console Output)

**Description**: Library for styled terminal output (not interactive)

**Technology**:
- Framework: [Rich](https://rich.readthedocs.io/)
- Language: Python 3.6+
- Dependencies: None (pure Python)
- Terminal: Works in any terminal

**Architecture**:
```python
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress

console = Console()

with Live(auto_refresh=True) as live:
    while agent.is_running():
        table = generate_status_table(agent)
        live.update(table)
        time.sleep(0.5)
```

**Pros**:
- ✅ Very simple to use
- ✅ Lightweight, no heavy dependencies
- ✅ Wide Python version support (3.6+)
- ✅ Great for progress bars, tables, syntax highlighting
- ✅ Works everywhere (fallback for old terminals)
- ✅ Minimal learning curve

**Cons**:
- ❌ Not truly interactive (no keyboard input handling)
- ❌ Limited to live-updating displays
- ❌ Can't have multiple scrollable panels
- ❌ No built-in layout management
- ❌ User can't navigate/inspect, only watch

**Best For**:
- Enhanced logging output
- Progress indicators
- Simple status displays
- Quick implementation (low risk)

**Effort**: Low (3-5 days for polished output)
**Risk**: Very Low (proven, stable library)
**Maintenance**: Very Low (simple code)

---

### Option 3: Hybrid (Rich + Textual)

**Description**: Use Rich for static output, Textual for interactive features

**Technology**:
- Framework: Rich (default), Textual (opt-in)
- Language: Python 3.8+
- Strategy: Progressive enhancement

**Architecture**:
```python
# Default: Rich output
if not args.tui:
    render_with_rich(agent)

# Opt-in: Textual dashboard
else:
    run_textual_dashboard(agent)
```

**Pros**:
- ✅ Best of both worlds
- ✅ Low barrier to entry (Rich is simple)
- ✅ Gradual migration path
- ✅ Users choose complexity level
- ✅ Rich works in CI/non-interactive environments

**Cons**:
- ❌ Maintain two rendering paths
- ❌ Feature parity challenges (Rich vs Textual)
- ❌ More testing surface area

**Best For**:
- Gradual rollout (ship Rich first, Textual later)
- Supporting both simple and advanced users
- Minimizing risk

**Effort**: Low + Medium (Rich first, Textual later)
**Risk**: Low (can ship Rich immediately)
**Maintenance**: Medium (two codepaths)

---

### Option 4: Web-Based TUI (Terminal in Browser)

**Description**: Use web technologies, accessed via browser

**Technology**:
- Frontend: React/Vue + xterm.js (terminal emulator)
- Backend: FastAPI/Flask (WebSocket for live updates)
- Deployment: Local server (http://localhost:8080) or cloud

**Architecture**:
```
┌─────────────┐      WebSocket      ┌──────────────┐
│   Browser   │ ←─────────────────→ │ Jetbox Agent │
│  (xterm.js) │     Live updates    │  (FastAPI)   │
└─────────────┘                     └──────────────┘
```

**Pros**:
- ✅ Rich interactivity (click, drag, copy/paste)
- ✅ Shareable (send link to teammate)
- ✅ Remote monitoring (access from phone)
- ✅ Familiar web dev tools (React DevTools, etc.)
- ✅ Easy to embed charts, images, videos
- ✅ Multi-user support (team dashboard)

**Cons**:
- ❌ Requires running a server
- ❌ More complex deployment
- ❌ Heavier resource usage (browser + server)
- ❌ Overkill for single-user local development
- ❌ Firewall/network issues possible

**Best For**:
- Team environments
- Remote/cloud deployments
- Stakeholder demos
- Future: SaaS offering

**Effort**: High (4-6 weeks)
**Risk**: Medium (network, browser compatibility)
**Maintenance**: High (frontend + backend + infrastructure)

**Note**: This is a "future vision" option, not Phase 1.

---

### Option 5: tmux/screen Integration

**Description**: Leverage existing terminal multiplexers

**Technology**:
- Framework: tmux/screen (external)
- Language: Shell scripts + Python
- Strategy: Auto-configure tmux layout

**Architecture**:
```bash
# Launch Jetbox in tmux with predefined layout
jetbox-tmux "Create calculator"

# Creates tmux session:
# ┌─────────────┬─────────────┐
# │   Logs      │  Files      │
# ├─────────────┴─────────────┤
# │   Status                  │
# └───────────────────────────┘
```

**Pros**:
- ✅ Leverages existing tools (many devs already use tmux)
- ✅ No new dependencies
- ✅ Familiar keybindings (for tmux users)
- ✅ Persistent sessions (detach/reattach)
- ✅ Very lightweight

**Cons**:
- ❌ Limited to tmux/screen users
- ❌ Less control over layout/styling
- ❌ Harder to customize
- ❌ Not discoverable for non-tmux users
- ❌ Platform-specific (tmux not on all systems)

**Best For**:
- Power users who live in tmux
- Minimal overhead
- Optional enhancement (not default)

**Effort**: Low (2-3 days for basic integration)
**Risk**: Low (optional feature)
**Maintenance**: Low (thin wrapper)

---

### Comparison Matrix

| Feature | Textual | Rich | Hybrid | Web | tmux |
|---------|---------|------|--------|-----|------|
| **Interactivity** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Visual Polish** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Development Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **Portability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Resource Usage** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Future-Proof** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## UX Design

### Information Architecture

#### Level 1: Core Status (Always Visible)

**What users need to know at a glance:**

```
┌─────────────────────────────────────────────────────┐
│ 🎯 Goal: Create blog system with CRUD operations   │
│ 🤖 Agent: task_executor (qwen3:14b)                │
│ ⏱️  Round: 12/50 │ Time: 3m 24s │ Status: Running │
│ [████████████░░░░░░░░░░░░░░] 24%                   │
└─────────────────────────────────────────────────────┘
```

**Elements**:
1. **Current Goal** - What the agent is trying to achieve
2. **Agent Identity** - Which agent, which model
3. **Progress** - Round count, elapsed time, % complete
4. **Status** - Running / Paused / Completed / Failed
5. **Progress Bar** - Visual progress indicator

**Design Principles**:
- Fits in ~5 lines (can see in small terminal split)
- Updates every second
- Clear visual hierarchy (goal most prominent)
- Color coding: green (running), yellow (paused), red (failed)

---

#### Level 2: Activity Feed (Scrollable)

**What users need to track progress:**

```
┌─ Activity Log ──────────────────────────────────────┐
│ 12:34:56 ℹ️  Starting task execution                │
│ 12:34:57 🔧 Tool: write_file(path=models.py)        │
│ 12:34:58 ✅ Created models.py (1.2 KB)              │
│ 12:34:59 🔧 Tool: run_bash(pytest test_models.py)   │
│ 12:35:02 ✅ Tests passed (3/3)                      │
│ 12:35:03 🔧 Tool: mark_complete(summary=...)        │
│ 12:35:04 🎉 Task completed successfully             │
│                                                      │
│ ▼ Scroll: 7 more entries ↓                          │
└──────────────────────────────────────────────────────┘
```

**Elements**:
1. **Timestamp** - When event occurred
2. **Icon** - Quick visual classification
3. **Event** - What happened (tool call, file created, etc.)
4. **Context** - Brief details (file path, command, etc.)

**Icon Key**:
- ℹ️ Info (agent starting, status updates)
- 🔧 Tool call (write_file, run_bash, etc.)
- ✅ Success (file created, tests passed)
- ⚠️ Warning (retry, non-critical error)
- ❌ Error (failure, exception)
- 🎉 Milestone (task completed, goal achieved)

**Interactions**:
- Auto-scroll to bottom (follow mode)
- Click line → expand details
- Search/filter by keyword
- Export to file

---

#### Level 3: Workspace View (File Tree)

**What users need to see what was created:**

```
┌─ Workspace (/tmp/blog_system) ──────────────────────┐
│                                                      │
│ 📁 architecture/                                    │
│   └─ 📄 blog-system.md                   (2.8 KB)  │
│ 📁 __pycache__/                                     │
│ 📄 models.py                              (1.2 KB)  │
│ 📄 blog_manager.py                        (2.1 KB)  │
│ 📄 json_persistence.py                    (1.5 KB)  │
│ 📄 test_blog_manager.py                   (0.8 KB)  │
│ 📄 posts.json                             (0.1 KB)  │
│                                                      │
│ 7 files, 8.5 KB total                               │
└──────────────────────────────────────────────────────┘
```

**Elements**:
1. **Directory tree** - Visual hierarchy
2. **File icons** - Quick identification (folder, code, data, etc.)
3. **File sizes** - Space awareness
4. **Summary stats** - Total files, total size

**Interactions**:
- Expand/collapse directories
- Click file → preview in side panel
- Highlight recently modified files
- Show git status (added, modified, deleted)

---

#### Level 4: Goal Hierarchy (For Orchestrator)

**What users need when agent delegates:**

```
┌─ Goal Tree ─────────────────────────────────────────┐
│                                                      │
│ 🎯 Blog System                              [24%]   │
│   ✅ Architecture (architect)              [100%]   │
│      ✅ Design system                               │
│      ✅ Create task breakdown                       │
│   🔄 Implementation (task_executor)         [15%]   │
│      ✅ Create models                               │
│      🔄 Create BlogManager                  [50%]   │
│      ⏸️  Write tests                        [ 0%]   │
│      ⏸️  Implement persistence              [ 0%]   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Elements**:
1. **Tree structure** - Parent → child goals
2. **Status icons** - Completed, in-progress, pending
3. **Agent attribution** - Which agent worked on what
4. **Progress %** - Per goal and overall

**Icon Key**:
- ✅ Completed
- 🔄 In progress
- ⏸️ Pending (not started)
- ❌ Failed
- ⏭️ Skipped

**Interactions**:
- Click goal → jump to logs for that goal
- Expand/collapse subtrees
- Filter: show only failures

---

#### Level 5: Context Inspector (On Demand)

**What power users need for debugging:**

```
┌─ Context Inspector: Round 12 ───────────────────────┐
│                                                      │
│ 📊 Stats                                            │
│   Total tokens: 45,234 / 128,000 (35%)             │
│   System prompt: 2,100 tokens                       │
│   Messages: 38,450 tokens                           │
│   Tool definitions: 4,684 tokens                    │
│                                                      │
│ 📝 System Prompt                                    │
│   # CONTEXT                                         │
│   You are a coding agent that implements...         │
│   [click to expand]                                 │
│                                                      │
│ 💬 Messages (12)                                    │
│   [1] user: Create blog system with...             │
│   [2] assistant: I'll start by creating models...  │
│   [3] assistant: <tool_call>write_file...          │
│   ...                                                │
│   [12] assistant: <tool_call>run_bash...           │
│                                                      │
│ 🔧 Tools (8)                                        │
│   write_file, read_file, list_dir, run_bash...     │
│                                                      │
│ [E]xport JSON │ [C]opy to clipboard │ [Q]uit       │
└──────────────────────────────────────────────────────┘
```

**Elements**:
1. **Token breakdown** - See what's using context
2. **System prompt** - What instructions agent has
3. **Message history** - Full conversation
4. **Tool definitions** - What tools are available

**Interactions**:
- Syntax highlighting (markdown, JSON, Python)
- Expand/collapse sections
- Search within context
- Export to file (for sharing, debugging)
- Copy to clipboard (for manual LLM testing)

---

### Interaction Patterns

#### Keyboard Controls (Standard Across All Views)

**Global**:
- `q` - Quit (graceful shutdown, save state)
- `Q` - Force quit (SIGTERM, no save)
- `?` - Help (show all keybindings)

**Navigation**:
- `Tab` / `Shift+Tab` - Cycle between panels
- `↑` `↓` `←` `→` - Navigate within panel
- `PgUp` `PgDn` - Scroll page up/down
- `Home` `End` - Jump to top/bottom

**Agent Control**:
- `p` - Pause (finish current round, then wait)
- `r` - Resume (continue from paused state)
- `s` - Step (execute one round, then pause)
- `k` - Kill (abort current task)

**Inspection**:
- `c` - Context inspector (view last prompt)
- `f` - File viewer (browse workspace)
- `l` - Logs (jump to log panel)
- `g` - Goals (jump to goal tree)

**Search/Filter**:
- `/` - Search logs (regex supported)
- `n` - Next search result
- `N` - Previous search result
- `Esc` - Clear search/filter

**Export**:
- `e` - Export current view (logs, context, stats)
- `E` - Export full session (all data)

**Advanced** (Power Users):
- `d` - Debug mode (show extra diagnostics)
- `t` - Toggle timing info (show per-round timing)
- `m` - Memory inspector (show token usage, model state)

---

#### Mouse Support (Optional, Textual Only)

**Interactions**:
- Click panel → focus that panel
- Scroll wheel → scroll within panel
- Click button → execute action
- Drag divider → resize panels

**Philosophy**:
- Keyboard-first (faster for power users)
- Mouse as fallback (discoverability for new users)

---

### Workflow Examples

#### Workflow 1: Simple Task Monitoring

**Scenario**: Developer runs "Create calculator with tests"

```
1. Launch: python agent.py "Create calculator"

2. TUI appears:
   ┌─────────────────────────────────────────────────┐
   │ 🎯 Create calculator with add, subtract, tests │
   │ 🤖 task_executor (qwen3:14b)                   │
   │ ⏱️  Round: 1/50 │ Time: 0:05 │ Status: Running│
   │ [##░░░░░░░░░░░░░░░░░░░░] 2%                    │
   └─────────────────────────────────────────────────┘

   12:00:01 ℹ️  Starting task execution
   12:00:02 🔧 write_file(calculator.py)

3. Developer switches to editor, works on other stuff

4. Glances back at terminal every few minutes:
   - Round 3/50 (6%)
   - Round 7/50 (14%)
   - ...

5. Sees completion:
   ┌─────────────────────────────────────────────────┐
   │ 🎉 Task completed successfully!                │
   │ ⏱️  Duration: 2m 34s │ Rounds: 8/50            │
   │                                                 │
   │ 📁 Files created:                              │
   │   ✅ calculator.py (245 bytes)                 │
   │   ✅ test_calculator.py (312 bytes)            │
   │                                                 │
   │ ✅ Tests: 4/4 passed                           │
   │ ✅ Linting: No errors                          │
   └─────────────────────────────────────────────────┘

6. Press 'q' to exit
```

**Key UX Principles**:
- Minimal distraction (compact status)
- At-a-glance progress (progress bar)
- Clear completion signal (celebration!)
- Actionable summary (what was created)

---

#### Workflow 2: Debugging Stuck Agent

**Scenario**: Agent stuck in loop, developer intervenes

```
1. Developer notices: Round 23/50, same error repeating

2. Press 'p' to pause:
   ┌─────────────────────────────────────────────────┐
   │ ⏸️  PAUSED - Round 23/50 paused after completion│
   │ Press 'r' to resume, 's' to step, 'q' to quit  │
   └─────────────────────────────────────────────────┘

3. Press 'c' to inspect context:
   ┌─ Context Inspector ─────────────────────────────┐
   │ Last 3 messages:                                │
   │ [21] assistant: <tool>run_bash(pytest...)      │
   │ [22] tool: Error: ModuleNotFoundError: models  │
   │ [23] assistant: <tool>run_bash(pytest...)      │
   │                                                 │
   │ 🔍 Analysis: Agent keeps retrying same command │
   │    without fixing the import error              │
   └─────────────────────────────────────────────────┘

4. Press 'f' to view files:
   ┌─ Workspace ─────────────────────────────────────┐
   │ 📄 models.py                                    │
   │ 📄 blog_manager.py                              │
   │ 📄 test_blog_manager.py                         │
   │                                                 │
   │ 🔍 Issue: test_blog_manager.py imports models  │
   │    but models.py has syntax error (line 15)    │
   └─────────────────────────────────────────────────┘

5. Developer options:
   a. Press 'q' to quit, fix manually
   b. Press 's' to step through next round
   c. Press 'e' to export context, file bug report

6. Developer chooses (a), fixes models.py manually
```

**Key UX Principles**:
- Easy interruption (pause with 'p')
- Deep inspection (context, files)
- Clear diagnosis (what's wrong)
- Flexible next steps (step, quit, export)

---

#### Workflow 3: Overnight Evaluation

**Scenario**: Researcher runs 50-task evaluation suite

```
1. Launch: python tests/orchestrator_l5_l7_eval.py --tui

2. Multi-task dashboard appears:
   ┌─ Evaluation Progress ───────────────────────────┐
   │ Suite: L5-L7 Orchestrator Eval                  │
   │ Progress: 12/50 completed (24%)                 │
   │ Success Rate: 10/12 (83%)                       │
   │ Elapsed: 2h 15m │ ETA: 6h 30m                   │
   │ [#####░░░░░░░░░░░░░░░] 24%                      │
   └─────────────────────────────────────────────────┘

   ┌─ Current Tasks ─────────────────────────────────┐
   │ 🔄 L5: blog_system (Round 18/50, 3m 12s)       │
   │ ⏸️  L5: ecommerce_cart (Queued)                │
   │ ⏸️  L5: user_auth (Queued)                     │
   └─────────────────────────────────────────────────┘

   ┌─ Recent Completions ────────────────────────────┐
   │ ✅ L5: todo_app (2m 45s, 12 rounds)            │
   │ ❌ L6: api_gateway (timeout, 60 rounds)        │
   │ ✅ L5: markdown_parser (1m 30s, 8 rounds)      │
   └─────────────────────────────────────────────────┘

3. Developer goes to sleep

4. Morning: checks TUI
   ┌─ Evaluation Complete ───────────────────────────┐
   │ 🎉 Finished at 06:23 AM                        │
   │ Results: 42/50 succeeded (84%)                 │
   │ Duration: 8h 45m                                │
   │                                                 │
   │ ✅ Passed: 42 tasks                            │
   │ ❌ Failed: 6 tasks (view details below)        │
   │ ⏱️  Timeout: 2 tasks                           │
   │                                                 │
   │ 💰 Total cost: $2.34                           │
   │ 📊 Avg time: 10m 30s per task                  │
   └─────────────────────────────────────────────────┘

5. Press 'e' to export results:
   Saved to: evaluation_results/run_20251118.json

6. Press 'f' to filter failures:
   ┌─ Failed Tasks ──────────────────────────────────┐
   │ ❌ L6: api_gateway (timeout after 60 rounds)   │
   │ ❌ L6: microservices (error: import failed)    │
   │ ❌ L7: distributed_cache (timeout)             │
   │ ❌ L5: blog_system (validation failed)         │
   │ ❌ L7: ml_pipeline (error: CUDA not available) │
   │ ❌ L6: graphql_api (validation failed)         │
   └─────────────────────────────────────────────────┘

7. Click on task → view full logs, inspect why it failed
```

**Key UX Principles**:
- High-level overview (progress, ETA)
- Live updates (current task status)
- Persistent display (doesn't require interaction)
- Post-completion analysis (filter failures, export)

---

## Visual Styles

### Style 1: Minimalist (Rich-based, Simple)

**Philosophy**: Less is more. Clean, readable, terminal-native look.

**Preview**:
```
┌─ Jetbox Agent ────────────────────────────────────────┐
│ Goal: Create calculator with add, subtract            │
│ Agent: task_executor │ Model: qwen3:14b               │
│ Round: 3/10 │ Time: 45s │ Tokens: 2.3k/128k          │
├───────────────────────────────────────────────────────┤
│ [####------] Creating calculator.py...                │
│                                                        │
│ [07:45:23] Tool: write_file(path=calculator.py)      │
│ [07:45:24] Created: calculator.py (245 bytes)        │
│ [07:45:25] Tool: run_bash(command=pytest test_cal...)│
│ [07:45:26] ✓ All tests passed (2/2)                  │
│ [07:45:27] Tool: mark_complete(summary=Created...)   │
│                                                        │
└───────────────────────────────────────────────────────┘
```

**Characteristics**:
- Single-line box drawing (┌─┐│└┘├┤)
- Monochrome or minimal color (green=good, red=bad)
- Dense information (no wasted space)
- Terminal-native fonts (no Unicode beyond box drawing)

**Colors**:
- Background: Terminal default
- Text: White/gray
- Success: Green
- Error: Red
- Warning: Yellow
- Info: Blue

**Pros**:
- ✅ Works everywhere (even old terminals)
- ✅ Easy to read at a glance
- ✅ Professional, no-nonsense
- ✅ Low resource usage

**Cons**:
- ❌ Less visual appeal
- ❌ Harder to differentiate sections
- ❌ Limited emoji/icon support

**Target Users**: Terminal purists, minimal setup, CI/CD environments

---

### Style 2: Modern/Material (Unicode, Emojis, Colors)

**Philosophy**: Embrace modern terminals. Rich visual feedback with emojis and colors.

**Preview**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  🤖 Jetbox                                             ┃
┃  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ┃
┃                                                        ┃
┃  📋 Create calculator with add, subtract               ┃
┃  🔧 task_executor  •  qwen3:14b  •  Round 3/10  •  45s ┃
┃                                                        ┃
┃  ▓▓▓▓▓▓░░░░░░░░░░░░░░ 30%                             ┃
┃                                                        ┃
┃  ✅ Architecture designed                              ┃
┃  🔄 Writing calculator.py...                           ┃
┃  ⏸️  Pending: Tests, Documentation                     ┃
┃                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  💬 Logs                              📁 Files         ┃
┃  ─────────────────────────────────    ───────────────  ┃
┃  12:34:56  Writing calculator.py      calculator.py   ┃
┃  12:34:57  Testing functions          test_calc.py    ┃
┃  12:34:58  ✓ All tests passed                         ┃
┃                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  💰 $0.001  •  📊 2.3k tokens  •  🎯 30% complete      ┃
┃  ⌨️  [P]ause  [C]ontext  [F]iles  [Q]uit               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Characteristics**:
- Double-line box drawing (┏━┓┃┗┛┣┫)
- Heavy Unicode use (emojis, symbols)
- Rich colors (syntax highlighting, status colors)
- Rounded corners, thick dividers

**Colors**:
- Background: Dark (#282a36 Dracula or #2e3440 Nord)
- Primary: Bright blue/cyan
- Success: Bright green (#50fa7b)
- Warning: Yellow
- Error: Red (#ff5555)
- Muted: Gray (#6272a4)

**Pros**:
- ✅ Visually appealing
- ✅ Emojis provide instant feedback
- ✅ Clear visual hierarchy
- ✅ Modern, polished feel

**Cons**:
- ❌ Requires Unicode support
- ❌ May look broken in old terminals
- ❌ Emoji rendering varies by OS/font

**Target Users**: Modern terminal users (iTerm2, Windows Terminal, Alacritty)

---

### Style 3: Dashboard/IDE-like (Textual, Multi-Panel)

**Philosophy**: Maximum information density. Split panels like tmux or VS Code.

**Preview**:
```
╔════════════════════════════════════════════════════════╗
║ File  View  Agent  Help                    🟢 Running ║
╠═══════════════╦════════════════════════════════════════╣
║ 📊 Status     ║  📝 Output                             ║
║ ───────────── ║  ────────────────────────────────────  ║
║               ║                                        ║
║ Goal:         ║  [12:34:56] Starting task execution    ║
║ Create calc   ║  [12:34:57] Tool: write_file(calc.py)  ║
║               ║  [12:34:58] Created calculator.py      ║
║ Agent:        ║  [12:34:59] Tool: run_bash(pytest...)  ║
║ task_executor ║  [12:35:00] Running: pytest test_cal...║
║               ║  ===== test session starts =====       ║
║ Model:        ║  collected 2 items                     ║
║ qwen3:14b     ║                                        ║
║               ║  test_calculator.py::test_add PASSED   ║
║ Progress:     ║  test_calculator.py::test_sub PASSED   ║
║ [####------]  ║                                        ║
║ 4/10 (40%)    ║  ===== 2 passed in 0.12s =====         ║
║               ║  [12:35:01] ✓ All tests passed         ║
║ Time: 0:02:34 ║  [12:35:02] Tool: mark_complete(...)   ║
║ Tokens: 12k   ║                                        ║
║               ║  ▼ Scroll: ↓ ↑ PgDn PgUp               ║
╠═══════════════╬════════════════════════════════════════╣
║ 📁 Workspace  ║  🔍 Context Preview                    ║
║ ───────────── ║  ────────────────────────────────────  ║
║               ║                                        ║
║ /tmp/calc/    ║  System: You are a coding agent...     ║
║ ├─ calc.py    ║  User: Create a calculator with add    ║
║ ├─ test_ca... ║  and subtract functions                ║
║ └─ .agent_... ║                                        ║
║               ║  Tools: write_file, run_bash...        ║
║ 2 files       ║                                        ║
║               ║  [C]ontext Full | [H]istory            ║
╠═══════════════╩════════════════════════════════════════╣
║ [P]ause [S]tep [R]esume [C]ontext [F]iles [Q]uit      ║
╚════════════════════════════════════════════════════════╝
```

**Characteristics**:
- 4-panel layout (Status, Output, Workspace, Context)
- Heavy use of dividers (═║╔╗╚╝╠╣)
- Resizable panels (drag dividers)
- Keyboard focus indicators

**Colors**:
- Panel borders: Muted blue/gray
- Active panel: Highlighted border
- Syntax highlighting in output/context panels
- Status indicators: color-coded dots (🟢🟡🔴)

**Pros**:
- ✅ Maximum information visible
- ✅ Familiar to IDE users
- ✅ Efficient use of screen space
- ✅ Great for large terminals

**Cons**:
- ❌ Overwhelming for simple tasks
- ❌ Requires large terminal (>100 cols)
- ❌ Complex to navigate for new users

**Target Users**: Power users, large screens, debugging sessions

---

### Style 4: Retro/Cyberpunk (ASCII Art, Neon Colors)

**Philosophy**: Fun, nostalgic, stands out. Heavy ASCII art.

**Preview**:
```
╔═══════════════════════════════════════════════════════╗
║  ██╗███████╗████████╗██████╗  ██████╗ ██╗  ██╗      ║
║  ██║██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗╚██╗██╔╝      ║
║  ██║█████╗     ██║   ██████╔╝██║   ██║ ╚███╔╝       ║
║  ██║██╔══╝     ██║   ██╔══██╗██║   ██║ ██╔██╗       ║
║  ██║███████╗   ██║   ██████╔╝╚██████╔╝██╔╝ ██╗      ║
║  ╚═╝╚══════╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝      ║
╠═══════════════════════════════════════════════════════╣
║  AGENT: task_executor            MODEL: qwen3:14b    ║
║  TASK:  Create calculator        TIME:  00:03:45     ║
║  ROUND: [████████████░░░░░░░░] 12/20 (60%)           ║
╠═══════════════════════════════════════════════════════╣
║  ▶ Writing calculator.py...........................  ║
║  ▶ Running tests...................................  ║
║  ✓ 2 tests passed                                    ║
║  ▶ Marking complete................................  ║
╠═══════════════════════════════════════════════════════╣
║  [P]AUSE  [R]ESUME  [C]ONTEXT  [Q]UIT                ║
╚═══════════════════════════════════════════════════════╝
```

**Characteristics**:
- ASCII art logo (large, pixelated)
- Neon/cyberpunk colors (cyan, magenta, green on black)
- "Hacker" aesthetic (monospace, retro feel)
- Animated elements (scanlines, glitch effects)

**Colors**:
- Background: Pure black (#000000)
- Primary: Neon cyan (#00ffff)
- Accent: Hot pink (#ff00ff)
- Success: Neon green (#00ff00)
- Error: Neon red (#ff0000)

**Pros**:
- ✅ Fun, memorable
- ✅ Great for demos/screenshots
- ✅ Nostalgic appeal

**Cons**:
- ❌ Less professional
- ❌ Hard to read for long periods
- ❌ May be distracting
- ❌ Not everyone's taste

**Target Users**: Hackers, streamers, demo/marketing use

---

### Style 5: Adaptive/Configurable

**Philosophy**: Let users choose. Detect terminal capabilities and adapt.

**Implementation**:
```yaml
# ~/.config/jetbox/tui.yaml
style: modern  # or: minimalist, dashboard, retro
color_scheme: dracula  # or: nord, solarized, github, custom
emoji: true  # Use emojis vs text
unicode: true  # Use Unicode box drawing vs ASCII
panels:
  layout: horizontal  # or: vertical, quad
  show_workspace: true
  show_context_preview: false
```

**Adaptive Logic**:
```python
def detect_terminal_capabilities():
    """Auto-detect terminal features."""
    return {
        "colors": supports_truecolor(),  # 24-bit color
        "unicode": supports_unicode(),   # UTF-8
        "emoji": supports_emoji(),       # Emoji rendering
        "size": get_terminal_size(),     # Cols × rows
    }

def choose_style(config, capabilities):
    """Pick best style for terminal."""
    if not capabilities["unicode"]:
        return "minimalist"
    elif capabilities["size"].cols < 100:
        return "modern"  # Compact
    else:
        return "dashboard"  # Full-featured
```

**Pros**:
- ✅ Works for everyone
- ✅ User control (power users customize)
- ✅ Graceful degradation (old terminals still work)
- ✅ Future-proof (new styles can be added)

**Cons**:
- ❌ More code to maintain (multiple renderers)
- ❌ Testing complexity (many combinations)
- ❌ Docs need to cover all options

**Target Users**: Everyone (default adapts, power users configure)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Ship basic Rich-based output (low risk, high value)

**Tasks**:
1. **Rich Integration** (2 days)
   - Add Rich dependency
   - Create `tui_rich.py` module
   - Replace print() calls with Rich console
   - Add progress bars, status tables

2. **Live Updates** (1 day)
   - Use Rich.live() for auto-updating display
   - Update status every 1 second
   - Show current round, time, tool call

3. **Color Coding** (1 day)
   - Green: success, completed
   - Red: errors, failures
   - Yellow: warnings, in-progress
   - Blue: info, tool calls

4. **Basic Controls** (1 day)
   - Ctrl+C: Graceful shutdown (save state)
   - Signal handling (SIGTERM, SIGINT)
   - Show "Press Ctrl+C to stop" hint

**Deliverable**:
- Enhanced console output (better than current print statements)
- Live-updating progress display
- Color-coded status
- Works in all terminals (fallback to no-color)

**Success Metrics**:
- Users can glance at terminal and understand status
- Progress is visible (round count, %)
- Errors stand out (red)

**Risk**: Very low (Rich is stable, well-documented)

---

### Phase 2: Interactive Dashboard (Weeks 2-3)

**Goal**: Full Textual dashboard with keyboard controls

**Tasks**:
1. **Textual Setup** (1 day)
   - Add Textual dependency
   - Create `tui_textual.py` module
   - Basic app shell with header/footer

2. **Layout** (2 days)
   - Implement 4-panel layout (status, logs, workspace, context)
   - Resizable panels
   - Tab key navigation

3. **Status Panel** (1 day)
   - Goal, agent, model, round, time
   - Progress bar
   - Color-coded status indicator

4. **Log Panel** (2 days)
   - Scrollable log viewer
   - Syntax highlighting (JSON, Python, errors)
   - Auto-scroll (follow mode)
   - Search/filter

5. **Workspace Panel** (1 day)
   - File tree (expandable directories)
   - File sizes, modification times
   - Click to preview file

6. **Context Panel** (1 day)
   - Show last prompt sent to LLM
   - Token count
   - Syntax highlighting
   - Export button

7. **Keyboard Controls** (1 day)
   - `p`: Pause agent
   - `r`: Resume agent
   - `s`: Step one round
   - `c`: Open context inspector
   - `q`: Quit

8. **Configuration** (1 day)
   - `--tui` flag (enable Textual dashboard)
   - Config file: `~/.config/jetbox/tui.yaml`
   - Theme selection (minimalist, modern, dashboard)

**Deliverable**:
- Full interactive TUI with 4 panels
- Keyboard navigation and controls
- Pause/resume/step functionality
- Context inspection
- Configurable themes

**Success Metrics**:
- Users can pause agent and inspect state
- Context is visible without opening JSON files
- Navigation is intuitive (Tab, arrows)

**Risk**: Medium (Textual is newer, but well-documented)

---

### Phase 3: Advanced Features (Weeks 4-5)

**Goal**: Multi-agent view, charts, exports

**Tasks**:
1. **Multi-Agent Dashboard** (3 days)
   - Grid view: show multiple agents
   - Tab: cycle between agents
   - Summary stats: X completed, Y in progress
   - Filter: show only failures

2. **Performance Charts** (2 days)
   - Token usage over time (line chart)
   - Rounds per task (bar chart)
   - Use Plotext (TUI charting library)

3. **Export Capabilities** (1 day)
   - Export logs: `e` → save to file
   - Export context: JSON dump
   - Export stats: CSV/JSON
   - Screenshot (terminal buffer capture)

4. **Notifications** (1 day)
   - Desktop notifications (on completion, error)
   - Webhook support (POST to URL on events)
   - Slack/Discord integration (optional)

5. **Remote Monitoring** (2 days)
   - Optional web server mode
   - Access via http://localhost:8080
   - WebSocket for live updates
   - Share read-only link

**Deliverable**:
- Multi-agent monitoring (evaluation runs)
- Performance visualization (charts)
- Export/sharing capabilities
- Notifications (desktop, webhook)

**Success Metrics**:
- Can monitor 10+ agents in single dashboard
- Export data for analysis (JSON, CSV)
- Get alerts on task completion

**Risk**: Medium-High (web server adds complexity)

---

### Rollout Strategy

**Week 1**: Ship Rich-based output
- Merge to main, release v0.1
- Gather feedback (is it better than print?)
- Iterate on color scheme, layout

**Week 2-3**: Alpha Textual dashboard
- Feature flag: `--tui-experimental`
- Limited users (early adopters)
- Gather feedback on keyboard controls, layout

**Week 4**: Beta Textual dashboard
- Enable by default: `--tui` (can disable with `--no-tui`)
- Broader testing
- Fix bugs, polish UX

**Week 5**: GA Textual dashboard
- Stable, documented
- Tutorial video
- Blog post announcement

**Week 6+**: Advanced features (as needed)
- Multi-agent view (for researchers)
- Web-based monitoring (for teams)
- Charts, exports (for analysis)

---

## Recommendation

### Recommended Approach: **Hybrid (Rich → Textual)**

**Rationale**:
1. **Low Risk**: Start with Rich (proven, simple)
2. **Quick Win**: Ship better output in 1 week
3. **Progressive Enhancement**: Add Textual later
4. **User Choice**: Power users opt into full dashboard
5. **Backward Compat**: Rich works in CI/old terminals

**Recommended Style**: **Modern/Material (Style 2)**

**Rationale**:
1. **Visual Appeal**: Emojis, colors, polished
2. **Information Density**: Good balance
3. **Modern Terminals**: Most users have Unicode support
4. **Fallback**: Auto-detect, use minimalist if needed

**Recommended Color Scheme**: **Dracula**

**Rationale**:
1. **Popular**: Widely used (VS Code, iTerm, etc.)
2. **High Contrast**: Easy to read
3. **Well-Tested**: Colors work well together
4. **Customizable**: Users can override via config

---

### Implementation Plan

**Week 1: Rich Foundation**
- Integrate Rich library
- Replace print statements
- Add progress bars, color coding
- Ship as default output

**Week 2-3: Textual Dashboard**
- Build 4-panel layout
- Keyboard controls (pause, resume, step)
- Context inspector
- Alpha release (`--tui-experimental`)

**Week 4: Polish & Beta**
- Fix bugs from alpha feedback
- Add file preview, search
- Performance optimization
- Beta release (`--tui` on by default)

**Week 5+: Advanced Features**
- Multi-agent view
- Charts, exports
- Web-based option (future)

---

## Open Questions

1. **Config Storage**: Where to store TUI preferences?
   - Option A: `~/.config/jetbox/tui.yaml` (XDG standard)
   - Option B: `.jetbox/tui.json` in project root (per-project)
   - Recommendation: A (global) + B (project override)

2. **Web UI**: Build now or defer?
   - Recommendation: Defer to Phase 3+ (after Textual solid)

3. **Accessibility**: Support screen readers?
   - Recommendation: Yes, use semantic HTML for web UI, announce state changes in terminal

4. **Telemetry**: Track which TUI features are used?
   - Recommendation: Optional, opt-in, anonymized (e.g., "90% of users use --tui flag")

5. **Themes**: Allow custom user themes?
   - Recommendation: Yes, define JSON schema for colors, let users override

---

## Success Metrics

### Short-Term (1 month)
- ✅ 80%+ users prefer new TUI over old print output
- ✅ <5 bugs reported (stability)
- ✅ <10% performance overhead (speed)

### Medium-Term (3 months)
- ✅ 50%+ users use `--tui` flag regularly
- ✅ 10+ community theme contributions
- ✅ Positive feedback on pause/resume functionality

### Long-Term (6 months)
- ✅ Web-based UI for remote monitoring
- ✅ Multi-agent dashboard used in evaluations
- ✅ Jetbox TUI cited as "best-in-class" vs competitors

---

## Appendix: Competitive Analysis

### Similar Tools

1. **Docker**:
   - TUI: None (logs only)
   - Strength: Simple, text-based
   - Weakness: No interactivity

2. **K9s (Kubernetes TUI)**:
   - TUI: Full-featured, keyboard-driven
   - Strength: Multi-resource view, live updates
   - Weakness: Steep learning curve

3. **Jupyter Notebooks**:
   - TUI: None (web-based)
   - Strength: Rich output (charts, images)
   - Weakness: Requires browser

4. **Aider (AI pair programmer)**:
   - TUI: Minimal (Rich-based)
   - Strength: Clean, readable logs
   - Weakness: No interactivity, no pause/resume

### Jetbox Differentiators

1. **Pause/Resume**: Unique among AI coding tools
2. **Context Inspection**: View exact prompt sent
3. **Multi-Agent**: Monitor delegated work
4. **Crash-Resilient**: Resume from any point

---

*Proposal created 2025-11-18 by Claude Code*
*For questions/feedback: See GitHub issues*
