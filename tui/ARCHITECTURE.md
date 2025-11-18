# TUI Architecture Diagram

## System Overview

```
┌──────────────────────────────────────────────────────┐
│                   base_agent.py                      │
│                                                      │
│  - Knows WHAT to display (events, status)           │
│  - Doesn't know HOW to display it                   │
│  - Uses DisplayInterface only                       │
│                                                      │
│  self.display.update_status(...)                    │
│  self.display.log_event(...)                        │
└───────────────────┬──────────────────────────────────┘
                    │
                    │ (uses interface)
                    ▼
┌──────────────────────────────────────────────────────┐
│              DisplayInterface (ABC)                  │
│                                                      │
│  Abstract methods:                                   │
│  - start()                                           │
│  - stop()                                            │
│  - update_status(...)                                │
│  - log_event(...)                                    │
│  - show_completion(...)                              │
│  - prompt_user(...)                                  │
└───────────────────┬──────────────────────────────────┘
                    │
                    │ (implemented by)
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
┌───────────────────┐   ┌────────────────────────┐
│  PlainDisplay     │   │  TextualDisplay        │
│                   │   │                        │
│  - print()        │   │  - Textual framework   │
│  - Works          │   │  - Interactive TUI     │
│    everywhere     │   │  - Keyboard controls   │
│  - CI/pipes       │   │  - Multi-panel         │
│  - Fallback       │   │  - Terminal only       │
└───────────────────┘   └────────────────────────┘
```

## Display Factory Logic

```
User runs agent
       │
       ▼
┌──────────────────────────┐
│   DisplayFactory.create() │
└──────────┬───────────────┘
           │
           ▼
    ┌─────────────┐
    │ Force mode? │
    └──────┬──────┘
           │
    ┌──────┴─────────────────────────┐
    │                                │
    ▼                                ▼
┌─────────┐                    ┌──────────┐
│ --tui   │                    │ --no-tui │
│ env var │                    │ env var  │
└────┬────┘                    └────┬─────┘
     │                              │
     ▼                              ▼
┌─────────────┐              ┌──────────────┐
│ Textual     │              │ Plain        │
│ Display     │              │ Display      │
└─────────────┘              └──────────────┘
           │
           │ (if force_mode=None)
           ▼
    ┌─────────────┐
    │ Auto-detect │
    └──────┬──────┘
           │
    ┌──────┴───────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌─────────────┐                      ┌──────────┐
│ Is a TTY?   │ NO                   │ Plain    │
│ Size OK?    │ ────────────────────>│ Display  │
│ TERM OK?    │                      └──────────┘
│ Textual OK? │
└──────┬──────┘
       │ YES (all checks pass)
       ▼
┌──────────────┐
│ Textual      │
│ Display      │
└──────────────┘
```

## Data Flow

```
Agent executes action
       │
       ▼
┌──────────────────────────┐
│ Create AgentEvent        │
│                          │
│ AgentEvent(              │
│   type=TOOL_CALL,        │
│   message="write_file",  │
│   details={...}          │
│ )                        │
└───────────┬──────────────┘
            │
            ▼
┌───────────────────────────┐
│ display.log_event(event)  │
└───────────┬───────────────┘
            │
            ▼
    ┌───────────────┐
    │ DisplayInterface │
    │ routes to...     │
    └───────┬──────────┘
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
PlainDisplay    TextualDisplay
    │                │
    ▼                ▼
print(...)      app.write_log(...)
                     │
                     ▼
              ┌──────────────┐
              │ Log widget   │
              │ updates      │
              └──────────────┘
```

## Event Types

```
EventType.INFO          → ℹ️  "Agent starting"
EventType.TOOL_CALL     → 🔧 "write_file(...)"
EventType.TOOL_RESULT   → →  "Created file"
EventType.SUCCESS       → ✅ "Tests passed"
EventType.WARNING       → ⚠️  "Retry attempt 2/3"
EventType.ERROR         → ❌ "File not found"
EventType.MILESTONE     → 🎉 "Task completed"
EventType.STATUS_UPDATE → 📊 "Round 5/50"
```

## Status Update Flow

```
Agent round completes
       │
       ▼
┌──────────────────────────────────────┐
│ display.update_status(               │
│   goal="Create calculator",          │
│   agent_name="task_executor",        │
│   model="qwen3:14b",                 │
│   current_round=5,                   │
│   max_rounds=50,                     │
│   elapsed_time=30.5,                 │
│   status="Running",                  │
│   tokens_used=2500,                  │
│   tokens_max=128000                  │
│ )                                    │
└───────────┬──────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ Plain or TUI? │
    └───────┬───────┘
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
┌─────────────┐  ┌──────────────────────┐
│ PlainDisplay │  │ TextualDisplay       │
├─────────────┤  ├──────────────────────┤
│ Prints:     │  │ Updates app state:   │
│             │  │                      │
│ [executor]  │  │ app.current_round=5  │
│ Round 5/50  │  │ app.status="Running" │
│ 0m30s       │  │                      │
│ Running     │  │ Triggers reactive    │
│ Tokens: 2%  │  │ watch_current_round()│
└─────────────┘  │                      │
                 │ Updates UI panel     │
                 └──────────────────────┘
```

## Pause/Resume Flow (TUI Only)

```
User presses 'p'
       │
       ▼
┌──────────────────────┐
│ action_pause()       │
│ self.paused = True   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Agent checks:        │
│ display.is_paused()  │
│ → Returns True       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│ Agent calls:             │
│ display.wait_if_paused() │
│ → Blocks in while loop   │
└──────────┬───────────────┘
           │
           │ (waits...)
           │
    User presses 'r'
           │
           ▼
┌──────────────────────┐
│ action_resume()      │
│ self.paused = False  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ wait_if_paused()     │
│ loop exits           │
│ Agent continues      │
└────────────────────── ┘
```

## File Structure

```
tui/
│
├── __init__.py
│   └─> Exports: DisplayFactory, DisplayInterface, AgentEvent, EventType
│
├── display_interface.py
│   ├─> EventType (enum)
│   ├─> AgentEvent (dataclass)
│   └─> DisplayInterface (ABC)
│       ├─> start()
│       ├─> stop()
│       ├─> update_status()
│       ├─> log_event()
│       ├─> show_completion()
│       ├─> prompt_user()
│       ├─> can_pause()
│       ├─> is_paused()
│       └─> wait_if_paused()
│
├── plain_display.py
│   └─> PlainDisplay (implements DisplayInterface)
│       ├─> Uses print()
│       ├─> Minimal dependencies
│       └─> Always works
│
├── textual_display.py
│   ├─> JetboxDashboard (Textual.App)
│   │   ├─> Header, Footer
│   │   ├─> Status panel
│   │   ├─> Log panel
│   │   └─> Keyboard bindings
│   │
│   └─> TextualDisplay (implements DisplayInterface)
│       ├─> Wraps JetboxDashboard
│       ├─> Interactive features
│       └─> Pause/resume support
│
├── display_factory.py
│   └─> DisplayFactory
│       ├─> create(force_mode)
│       ├─> _auto_detect()
│       └─> _create_textual()
│
└── README.md, INTEGRATION_GUIDE.md, ARCHITECTURE.md
```

## Key Interfaces

### DisplayInterface (contract)
```python
class DisplayInterface(ABC):
    """All displays must implement these methods."""

    @abstractmethod
    def start(self) -> None:
        """Initialize display."""

    @abstractmethod
    def stop(self) -> None:
        """Cleanup display."""

    @abstractmethod
    def update_status(self, goal, agent_name, ...) -> None:
        """Update main status."""

    @abstractmethod
    def log_event(self, event: AgentEvent) -> None:
        """Log an event."""

    @abstractmethod
    def show_completion(self, success, summary, ...) -> None:
        """Show completion screen."""

    @abstractmethod
    def prompt_user(self, question: str) -> str:
        """Get user input."""

    # Optional (interactive only)
    def can_pause(self) -> bool: return False
    def is_paused(self) -> bool: return False
    def wait_if_paused(self) -> None: pass
```

### AgentEvent (data structure)
```python
@dataclass
class AgentEvent:
    """Universal event format."""
    type: EventType                    # Required
    message: str                       # Required
    details: Optional[dict] = None     # Optional
    timestamp: Optional[str] = None    # Optional
```

## Responsibilities

### base_agent.py
- ✅ Execute agent logic
- ✅ Call display methods at right times
- ❌ Know how to format output
- ❌ Know about Textual/Rich

### DisplayInterface
- ✅ Define contract (what methods)
- ❌ Implement anything

### PlainDisplay
- ✅ Simple print() output
- ✅ Work everywhere
- ❌ Interactive features

### TextualDisplay
- ✅ Rich interactive UI
- ✅ Keyboard controls
- ✅ Pause/resume
- ❌ Work in pipes/non-TTY

### DisplayFactory
- ✅ Choose right display
- ✅ Auto-detect environment
- ✅ Respect user preference
- ❌ Know implementation details

---

*Clean separation of concerns - each component has one job*
