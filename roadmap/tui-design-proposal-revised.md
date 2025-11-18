# Jetbox TUI Design Proposal (Revised)

**Date**: 2025-11-18
**Focus**: Terminal-only approach for local enthusiasts
**Status**: Tech spike complete - verified approaches only

---

## User Persona: Local Enthusiast

**Profile**:
- Runs agents locally on GPU (Ollama, not cloud)
- Values transparency and understanding of system state
- Enjoys tinkering with settings and configurations
- Wants full control, not black-box automation
- Terminal is primary interface (not web/GUI)

**Core Needs**:
1. **Constant feedback** - Know what agent is doing at all times
2. **Transparency** - See exact prompts, context, decisions
3. **Control dials** - Adjust verbosity from minimal to verbose
4. **Responsiveness** - Updates feel live, not batch-logged
5. **Inspectability** - Drill down into any round's context
6. **No surprises** - Clear signals when agent needs intervention

**Mental Model**:
- Think: `htop` for system monitoring
- Think: `docker stats` for live container metrics
- Think: `kubectl get pods --watch` for Kubernetes
- NOT: passive log tailing (`tail -f`)
- NOT: batch job submission (cron)

**Workflow Philosophy**:
- Agent runs in foreground (not background daemon)
- Terminal window dedicated to agent monitoring
- Can pause/inspect/resume at any time
- Feels like driving a car, not sending mail

**Detail Control Spectrum**:
```
Minimal          Default         Verbose          Debug
┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐
│ ✓ 3 │ →      │ Round│ →      │Tool │ →      │ Full│
│tasks│         │12/50│         │Calls│         │JSON │
└─────┘         └─────┘         └─────┘         └─────┘
 Status only    +Progress       +Logs           +Context
```

**Example Use Cases**:

**Use Case 1: Background Monitoring**
- Agent running calculator task
- Terminal in bottom split of tmux
- Glances at progress every 30 seconds
- **Needs**: Single-line status (round, time, current action)

**Use Case 2: Debugging Stuck Agent**
- Agent looping on same error
- Full terminal focus on agent
- Wants to see last 3 LLM responses
- **Needs**: Scrollable log, context preview, pause button

**Use Case 3: Learning How It Works**
- First-time user exploring capabilities
- Wants to understand decision-making
- Reads prompts, sees tool choices
- **Needs**: Annotated output explaining "why agent did X"

**Use Case 4: Overnight Eval Run**
- Running 50 tasks while sleeping
- Checks terminal in morning
- Wants summary, not full logs
- **Needs**: Completion report, failed task list, export to file

---

## Technical Approach: Terminal-Only (Verified)

### Tech Spike Findings

**Tested Approaches:**
1. ✅ **Rich.Live()** - Works for inline updates (VERIFIED)
2. ✅ **Textual** - Works for full TUI apps (VERIFIED)
3. ❌ **Past failures** - Identified root cause (mixing `print()` with Rich)

**Key Discovery**: Previous TUI attempts failed because:
- Jetbox has 82+ `print()` calls in base_agent.py
- Mixing `print()` with `Rich.Live()` breaks inline updates
- Must use `live.console.print()` exclusively

**Terminal Requirements**:
- TTY (not piped/redirected)
- ANSI escape code support (most modern terminals)
- Cursor positioning support (VT100+)

---

### Option 1: Rich.Live() - Inline Status Updates

**What It Is**: Single area of terminal that updates in-place

**How It Works (Verified)**:
```python
from rich.live import Live
from rich.table import Table
import time

def make_status_table(agent):
    """Generate status display."""
    table = Table(title="Jetbox Agent")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Goal", agent.goal[:50])
    table.add_row("Round", f"{agent.current_round}/{agent.max_rounds}")
    table.add_row("Time", format_duration(agent.elapsed_time))
    table.add_row("Status", agent.status)

    return table

# Main loop
with Live(make_status_table(agent), refresh_per_second=2) as live:
    while agent.is_running():
        agent.step()  # Execute one round
        live.update(make_status_table(agent))  # Refresh display
```

**Technical Details**:
- Uses ANSI escape codes: `\033[<row>;<col>H` (cursor position)
- Clears old content: `\033[K` (erase to end of line)
- Updates at configurable rate (default 4/sec, recommend 2/sec)
- Falls back gracefully: if not TTY, appends instead

**What You See**:
```
     Jetbox Agent
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field  ┃ Value                     ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Goal   │ Create calculator with... │
│ Round  │ 12/50                     │
│ Time   │ 3m 24s                    │
│ Status │ Running write_file(...)   │
└────────┴───────────────────────────┘
```

**Pros**:
- ✅ Actually works (tested and verified)
- ✅ Lightweight (just Rich dependency)
- ✅ Graceful fallback (works in CI/pipes)
- ✅ Simple to implement (< 100 lines of code)
- ✅ No learning curve for devs

**Cons**:
- ❌ Single update area (can't have scrollable log + status)
- ❌ No keyboard input (can't pause/resume interactively)
- ❌ Limited layout (one renderable)

**Best For**: Phase 1 - Better than current print() output

**Estimated Effort**: 2-3 days

**Risk**: Very low (proven approach)

---

### Option 2: Textual - Full Interactive Dashboard

**What It Is**: Complete TUI framework with panels, keyboard input, mouse support

**How It Works (Verified)**:
```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Log
from textual.containers import Container
from textual.reactive import reactive

class JetboxDashboard(App):
    """Interactive agent monitor."""

    CSS = """
    #status { height: 5; background: $boost; }
    #logs { height: 1fr; }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("r", "resume", "Resume"),
        ("c", "context", "Context"),
        ("q", "quit", "Quit"),
    ]

    round_num = reactive(0)  # Auto-updates UI when changed

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status")
        yield Log(id="logs")
        yield Footer()

    def on_mount(self) -> None:
        """Start agent when app loads."""
        self.agent = create_agent()
        self.run_agent_loop()

    def run_agent_loop(self) -> None:
        """Execute agent and update UI."""
        log_widget = self.query_one("#logs", Log)

        while self.agent.is_running():
            # Execute round
            result = self.agent.step()

            # Update UI
            self.round_num += 1  # Triggers watch_round_num()
            log_widget.write_line(f"Tool: {result['tool_name']}")

    def watch_round_num(self, new_value: int) -> None:
        """Called automatically when round_num changes."""
        status = self.query_one("#status", Static)
        status.update(f"Round: {new_value}/{self.agent.max_rounds}")

    def action_pause(self) -> None:
        """Called when user presses 'p'."""
        self.agent.pause()
        self.notify("Agent paused")
```

**Technical Details**:
- Uses alternate screen buffer (fullscreen mode)
- Restores terminal on exit
- Event-driven architecture (reactive)
- Async/await support (non-blocking UI)
- CSS-like styling system

**What You See**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  JetboxApp                   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Goal: Create calculator                      ┃
┃ Round: 12/50 │ Time: 3m24s │ Status: Running ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 12:34:56 Tool: write_file(calculator.py)    ┃
┃ 12:34:57 Created: calculator.py (245 bytes) ┃
┃ 12:34:58 Tool: run_bash(pytest tests/)      ┃
┃ 12:34:59 Output: ===== 2 passed in 0.1s === ┃
┃ 12:35:00 Tool: mark_complete(summary=...)   ┃
┃                                              ┃
┃ [Scrollable log area continues...]          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ p Pause │ r Resume │ c Context │ q Quit     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Pros**:
- ✅ Full interactivity (keyboard, mouse)
- ✅ Multiple panels (status + logs + workspace)
- ✅ Scrollable content (review history)
- ✅ Reactive updates (change data, UI updates)
- ✅ Professional look (polished, modern)

**Cons**:
- ❌ Heavier dependency (Textual + Rich)
- ❌ Fullscreen only (takes over terminal)
- ❌ More complex code (~500 lines)
- ❌ Learning curve (new paradigm)

**Best For**: Phase 2 - Full-featured interactive mode

**Estimated Effort**: 1-2 weeks

**Risk**: Medium (more complex, but documented)

---

### Option 3: Hybrid - Progressive Enhancement

**What It Is**: Start with Rich, add Textual later as opt-in

**Architecture**:
```python
class AgentDisplay:
    """Abstract display interface."""
    def update_status(self, agent):
        raise NotImplementedError

    def log_event(self, event):
        raise NotImplementedError

class RichDisplay(AgentDisplay):
    """Simple inline status."""
    def __init__(self):
        self.live = Live(...)

    def update_status(self, agent):
        self.live.update(make_table(agent))

    def log_event(self, event):
        self.live.console.print(event)

class TextualDisplay(AgentDisplay):
    """Full interactive dashboard."""
    def __init__(self):
        self.app = JetboxDashboard()

    def update_status(self, agent):
        self.app.round_num = agent.current_round

    def log_event(self, event):
        self.app.write_log(event)

# Usage
if args.tui:
    display = TextualDisplay()  # Opt-in
else:
    display = RichDisplay()  # Default

agent.run(display=display)
```

**Rollout Plan**:
1. **Week 1**: Ship RichDisplay (replaces all print() calls)
2. **Week 2-3**: Build TextualDisplay (feature parity)
3. **Week 4**: Beta test TextualDisplay with `--tui` flag
4. **Week 5**: Make TextualDisplay default (keep Rich as fallback)

**Pros**:
- ✅ Low risk (ship value early)
- ✅ Gradual migration (users adapt slowly)
- ✅ Fallback option (Rich always works)
- ✅ Testing time (polish Textual before default)

**Cons**:
- ❌ Maintain two codepaths
- ❌ Feature parity challenge
- ❌ More testing surface

**Best For**: Recommended approach - minimize risk

---

### Why Past Attempts Failed: Root Cause Analysis

**The Problem**:
```python
# ❌ WRONG - This breaks inline updates
with Live(status_table) as live:
    while agent.running:
        print(f"Debug: round {i}")  # BREAKS IT!
        live.update(status_table)
```

**Why It Breaks**:
1. `print()` writes to stdout directly
2. `Live()` tracks cursor position
3. `print()` moves cursor down (appends)
4. `Live()` loses track of where it was
5. Next update appends instead of replacing

**The Fix**:
```python
# ✅ CORRECT - Use live.console.print()
with Live(status_table) as live:
    while agent.running:
        live.console.print(f"Debug: round {i}")  # Works!
        live.update(status_table)
```

**Required Changes to Jetbox**:
1. Replace all `print()` with `display.log()` or `live.console.print()`
2. Pass `console` instance to all methods that log
3. Never use `print()` in agent code (only in tests/scripts)

**Affected Files**:
- `base_agent.py`: 82 instances of `print()`
- `llm_utils.py`: Unknown count
- `behaviors/*.py`: Unknown count

**Refactor Strategy**:
1. Create `AgentConsole` wrapper class
2. Replace `print(...)` → `console.log(...)`
3. In Rich mode: routes to `live.console.print()`
4. In Textual mode: routes to `Log` widget
5. In test mode: routes to stdout (no Live())

---

### Terminal Capability Detection

**Auto-Detect Strategy**:
```python
def detect_display_mode():
    """Choose best display mode for terminal."""

    # 1. Check if TTY
    if not sys.stdout.isatty():
        return "plain"  # Piped/redirected - no fancy output

    # 2. Check terminal size
    try:
        size = os.get_terminal_size()
        if size.columns < 80:
            return "plain"  # Too narrow for dashboard
    except OSError:
        return "plain"

    # 3. Check TERM variable
    term = os.environ.get("TERM", "dumb")
    if term in ["dumb", "unknown"]:
        return "plain"

    # 4. Check for Rich support
    from rich.console import Console
    console = Console()
    if not console.is_terminal:
        return "plain"

    # 5. Check user preference
    if os.environ.get("JETBOX_TUI") == "textual":
        return "textual"
    elif os.environ.get("JETBOX_TUI") == "rich":
        return "rich"
    elif os.environ.get("JETBOX_TUI") == "none":
        return "plain"

    # 6. Default based on terminal size
    if size.columns >= 120 and size.lines >= 24:
        return "textual"  # Big terminal - full dashboard
    else:
        return "rich"  # Smaller - compact status

# Usage
mode = detect_display_mode()
display = {
    "plain": PlainDisplay(),
    "rich": RichDisplay(),
    "textual": TextualDisplay(),
}[mode]
```

**Fallback Hierarchy**:
```
Textual (best)
    ↓ (terminal too small)
Rich.Live
    ↓ (not a TTY)
Plain print()
```

---

### Verbosity Control

**Detail Levels**:
```python
class VerbosityLevel(Enum):
    MINIMAL = 1   # Status only, no logs
    DEFAULT = 2   # Status + major events
    VERBOSE = 3   # Status + all tool calls
    DEBUG = 4     # Status + logs + context preview

# Configure via flag or env var
agent.run(verbosity=VerbosityLevel.VERBOSE)
# or
export JETBOX_VERBOSITY=verbose
```

**What Each Level Shows**:

**MINIMAL** (for background monitoring):
```
[████████░░░░░░░░░░░░] 40% - Round 12/50 - 3m24s - Running
```

**DEFAULT** (recommended):
```
┌─ Jetbox Agent ───────────────────────────────────┐
│ Goal: Create calculator with tests              │
│ Round: 12/50 │ Time: 3m24s │ Status: Running    │
│ [████████░░░░░░░░░░░░] 40%                      │
├──────────────────────────────────────────────────┤
│ ✅ Created calculator.py                         │
│ 🔧 Running tests...                              │
└──────────────────────────────────────────────────┘
```

**VERBOSE** (for learning/debugging):
```
┌─ Jetbox Agent ───────────────────────────────────┐
│ Goal: Create calculator with tests              │
│ Round: 12/50 │ Time: 3m24s │ Status: Running    │
│ [████████░░░░░░░░░░░░] 40%                      │
├──────────────────────────────────────────────────┤
│ 12:34:56 🔧 write_file(path=calculator.py)      │
│          ✅ Success (245 bytes)                  │
│ 12:34:57 🔧 run_bash(command=pytest tests/)     │
│          ✅ 2 passed in 0.1s                     │
│ 12:34:58 🔧 mark_complete(summary=Created...)   │
└──────────────────────────────────────────────────┘
```

**DEBUG** (for development):
```
┌─ Jetbox Agent ───────────────────────────────────┐
│ Goal: Create calculator with tests              │
│ Round: 12/50 │ Time: 3m24s │ Tokens: 45k/128k  │
│ [████████░░░░░░░░░░░░] 40%                      │
├──────────────────────────────────────────────────┤
│ 12:34:56 🔧 write_file(path=calculator.py)      │
│          ✅ Success (245 bytes)                  │
│          Context: 2,134 tokens                   │
│          LLM latency: 1.2s                       │
│ 12:34:57 🔧 run_bash(command=pytest tests/)     │
│          ✅ 2 passed in 0.1s                     │
│                                                  │
│ [Press 'c' to view full context]                │
└──────────────────────────────────────────────────┘
```

**Runtime Toggle** (Textual only):
- `+` / `-` keys: increase/decrease verbosity
- `d` key: toggle debug mode
- `m` key: toggle minimal mode

---

### Implementation Checklist

**Phase 1: Rich.Live() (Week 1)**
- [ ] Create `AgentConsole` abstraction
- [ ] Replace all `print()` in base_agent.py
- [ ] Implement `RichDisplay` class
- [ ] Add status table rendering
- [ ] Test in TTY and non-TTY environments
- [ ] Add verbosity levels (minimal, default, verbose)
- [ ] Documentation: How to read the status display

**Phase 2: Textual Dashboard (Weeks 2-3)**
- [ ] Create `TextualDisplay` class
- [ ] Implement 3-panel layout (status, logs, footer)
- [ ] Add keyboard bindings (p/r/s/c/q)
- [ ] Implement pause/resume functionality
- [ ] Add context inspector panel (on-demand)
- [ ] Test with real agent runs
- [ ] Beta release with `--tui` flag

**Phase 3: Polish & Rollout (Week 4)**
- [ ] Fix bugs from beta testing
- [ ] Add verbosity runtime toggle (+/- keys)
- [ ] Implement auto-detection (terminal size)
- [ ] Add config file support (`~/.config/jetbox/tui.yaml`)
- [ ] Write comprehensive docs (with screenshots)
- [ ] Make Textual default, Rich as fallback
- [ ] Release announcement (blog post, demo video)

---

### Success Criteria

**Week 1 (Rich)**:
- ✅ No `print()` calls in main agent loop
- ✅ Status updates in-place (verified with eyes)
- ✅ Fallback works in pipes/redirects
- ✅ 2+ verbosity levels implemented

**Week 2-3 (Textual)**:
- ✅ Pause/resume works without crashes
- ✅ Context inspector shows last prompt
- ✅ Logs are scrollable (review history)
- ✅ Keyboard bindings feel natural

**Week 4 (GA)**:
- ✅ 90%+ users prefer new TUI over old output
- ✅ <5 bugs reported in first week
- ✅ Docs complete (quickstart + reference)
- ✅ Positive feedback on transparency/control

---

### Open Questions

1. **Color Scheme**: Dracula? Nord? User-configurable?
   - Recommendation: Start with Dracula (most popular), add config later

2. **Emoji Usage**: Always show or optional?
   - Recommendation: Default ON, can disable with `--no-emoji`

3. **Log Retention**: How many lines to keep in memory?
   - Recommendation: Last 1000 lines (sufficient for scrollback)

4. **Export Format**: What format for log export?
   - Recommendation: Plain text (`.log`), Markdown (`.md`), JSON (`.json`)

5. **Multi-Agent View**: Single terminal or separate terminals?
   - Recommendation: Defer to Phase 3, start with single-agent focus

---

*Revised proposal based on verified technical research - 2025-11-18*
