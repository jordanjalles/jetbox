# Rich vs Textual: What's the Difference?

**TL;DR**:
- **Rich** = Library for pretty terminal OUTPUT (colors, tables, progress bars)
- **Textual** = Full TUI FRAMEWORK for building interactive APPS (like a terminal GUI)

---

## Rich: Enhanced Terminal Output

**What it is**: A library for rendering styled content to the terminal.

**Think of it as**: "Better `print()`" - makes terminal output beautiful and structured.

**Interactivity**: **NONE** - it's output-only, you can't click or press keys.

**Example - Basic Rich**:
```python
from rich.console import Console

console = Console()

# Pretty colored output
console.print("[bold green]✓ Success![/bold green]")
console.print("[red]Error:[/red] File not found")

# Tables
from rich.table import Table
table = Table(title="Files")
table.add_column("Name", style="cyan")
table.add_column("Size", style="magenta")
table.add_row("main.py", "2.3 KB")
table.add_row("test.py", "1.1 KB")
console.print(table)

# Progress bars
from rich.progress import track
for i in track(range(100), description="Processing..."):
    time.sleep(0.01)
```

**Output (static, appends down the screen)**:
```
✓ Success!
Error: File not found

    Files
┏━━━━━━━━━┳━━━━━━━━┓
┃ Name    ┃ Size   ┃
┡━━━━━━━━━╇━━━━━━━━┩
│ main.py │ 2.3 KB │
│ test.py │ 1.1 KB │
└─────────┴────────┘

Processing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

---

## Rich.Live(): The Bridge

**What it is**: A special Rich feature that updates ONE area in-place.

**Think of it as**: A designated "status box" that refreshes without appending.

**Interactivity**: Still **NONE** - just updates automatically, no user input.

**Example - Rich.Live()**:
```python
from rich.live import Live
from rich.table import Table
import time

def make_status(count):
    table = Table(title="Live Status")
    table.add_column("Counter")
    table.add_column("Status")
    table.add_row(str(count), "Running..." if count < 10 else "Done!")
    return table

# This updates IN PLACE, not appending
with Live(make_status(0), refresh_per_second=4) as live:
    for i in range(1, 15):
        time.sleep(0.5)
        live.update(make_status(i))  # Replaces old table
```

**Output (updates in-place, no scrolling)**:
```
      Live Status
┏━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Counter ┃ Status     ┃
┡━━━━━━━━━╇━━━━━━━━━━━━┩
│ 14      │ Done!      │  ← This number changes in-place!
└─────────┴────────────┘
```

**Limitation**: You can only update ONE renderable (one table, one panel, etc.).

**Use case**: Status displays, progress monitors, dashboard-style output.

---

## Textual: Full TUI Framework

**What it is**: A framework for building complete terminal user interfaces (like `htop`, `vim`, `nano`).

**Think of it as**: "React for terminals" - build interactive apps with components.

**Interactivity**: **FULL** - keyboard input, mouse clicks, scrollable panels, buttons.

**Example - Textual App**:
```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button, Log
from textual.containers import Container

class MyApp(App):
    """A simple Textual app with multiple interactive panels."""

    CSS = """
    #status { height: 3; background: $boost; }
    #logs { height: 1fr; }
    Button { margin: 1; }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header()
        yield Static("Status: Running", id="status")
        yield Log(id="logs")
        yield Container(
            Button("Pause", id="pause-btn"),
            Button("Resume", id="resume-btn"),
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.set_interval(1, self.update_counter)
        self.counter = 0

    def update_counter(self) -> None:
        """Update display every second."""
        self.counter += 1
        status = self.query_one("#status", Static)
        status.update(f"Status: Running (count: {self.counter})")

        logs = self.query_one("#logs", Log)
        logs.write_line(f"[{self.counter}] Heartbeat")

    def action_pause(self) -> None:
        """Called when user presses 'p'."""
        self.notify("Paused!")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when button clicked."""
        if event.button.id == "pause-btn":
            self.notify("Pause button clicked!")

if __name__ == "__main__":
    app = MyApp()
    app.run()
```

**Output (full-screen, interactive app)**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   MyApp                    ┃  ← Header
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Status: Running (count: 5)                ┃  ← Status panel
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ [1] Heartbeat                             ┃
┃ [2] Heartbeat                             ┃  ← Scrollable log
┃ [3] Heartbeat                             ┃     (can scroll up/down)
┃ [4] Heartbeat                             ┃
┃ [5] Heartbeat                             ┃
┃                                            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌────────┐  ┌────────┐                   ┃  ← Clickable buttons
┃  │ Pause  │  │ Resume │                   ┃
┃  └────────┘  └────────┘                   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ p Pause │ q Quit                          ┃  ← Footer (shortcuts)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Capabilities**:
- Multiple panels (status, logs, file tree, etc.)
- Keyboard shortcuts (press 'p' → calls action_pause())
- Mouse input (click buttons)
- Scrolling (up/down arrows in log panel)
- Reactive updates (change data, UI auto-updates)
- Layout system (CSS-like styling)
- Takes over entire terminal (fullscreen)

**Use case**: Complex interactive tools (file managers, debuggers, monitoring dashboards).

---

## Comparison Table

| Feature | Rich | Rich.Live() | Textual |
|---------|------|-------------|---------|
| **Output style** | Appending | In-place updates | Full-screen app |
| **Keyboard input** | ❌ None | ❌ None | ✅ Full support |
| **Mouse input** | ❌ None | ❌ None | ✅ Full support |
| **Multiple panels** | ❌ No | ❌ Only one | ✅ Yes |
| **Scrollable content** | ❌ No | ❌ No | ✅ Yes |
| **Screen takeover** | ❌ No | ❌ No | ✅ Yes (alternate screen) |
| **Learning curve** | Very easy | Easy | Moderate |
| **Code complexity** | ~10 lines | ~20 lines | ~100 lines |
| **Best for** | Logs, output | Status displays | Interactive apps |

---

## Analogy

**Rich**: Like using `printf()` with colors and tables.
```c
printf("\033[32m✓ Success\033[0m\n");  // Green text
```

**Rich.Live()**: Like updating a specific area of the screen.
```c
printf("\033[5;10H");  // Move cursor to row 5, col 10
printf("Count: %d", counter);  // Overwrite old value
```

**Textual**: Like building a GUI app but in the terminal.
```python
# Full event-driven framework
# Components, layouts, state management
# Like React/Vue but for terminals
```

---

## Real-World Examples

**Tools that use Rich**:
- `pip` (progress bars)
- `httpie` (colored HTTP responses)
- `poetry` (dependency installation output)
- **Style**: Enhanced logs, prettier output

**Tools that use Textual**:
- `gh dash` (GitHub dashboard)
- `posting` (Postman-like TUI)
- `dolphie` (MySQL monitoring)
- **Style**: Full interactive applications

**Tools that DON'T need either**:
- `ls` (just text output)
- `grep` (search results)
- **Style**: Plain text is fine

---

## When to Use What for Jetbox

### Use Rich.Live() when:
- ✅ You want better output than `print()`
- ✅ You need live status updates (round count, progress)
- ✅ Terminal might be redirected to a file (need fallback)
- ✅ Simple, low-risk implementation
- ✅ Phase 1 / MVP

**Example**: Agent running in background, quick status glance.
```
[████████░░░░░░░░░░░░] 40% - Round 12/50 - Running
```

### Use Textual when:
- ✅ You want full interactivity (pause, resume, inspect)
- ✅ You need multiple panels (logs + status + files)
- ✅ Users want to explore context, scroll through history
- ✅ Terminal is dedicated to agent (not background)
- ✅ Phase 2 / Full-featured

**Example**: Agent debugging session, need deep inspection.
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Status │ Logs │ Files │ Context           ┃
┃ [Interactive dashboard with keyboard]     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## The Hybrid Approach (Recommended)

**Phase 1** (Week 1): Rich.Live()
- Replace all `print()` calls
- Add in-place status updates
- Works everywhere (graceful fallback)
- Low risk, high value

**Phase 2** (Weeks 2-3): Textual (opt-in)
- Build full interactive dashboard
- Enable with `--tui` flag
- Beta test with power users
- Polish based on feedback

**Phase 3** (Week 4+): Make Textual default
- Most users get interactive dashboard
- Rich.Live() as fallback (TTY detection)
- Best of both worlds

---

## Key Takeaway

**Rich = Prettier output library**
- Use for: Making terminal output beautiful
- No interactivity
- Appends (or updates one area with Live())

**Textual = Terminal app framework**
- Use for: Building interactive terminal apps
- Full keyboard/mouse support
- Takes over screen (like vim, htop)

**For Jetbox**: Start with Rich.Live() (simple), add Textual later (powerful).

---

*2025-11-18 - Explanation based on verified testing*
