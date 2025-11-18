"""
Textual Display - Interactive TUI dashboard.

This is the main TUI implementation using the Textual framework.

Features:
- Multi-panel layout (status, logs, file tree)
- Keyboard controls (p/r/s/c/q)
- Pause/resume functionality
- Context inspection
- Scrollable log history
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Log, Tree
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from textual.binding import Binding
import time
from datetime import datetime
from pathlib import Path

from .display_interface import DisplayInterface, AgentEvent, EventType


class JetboxDashboard(App):
    """
    Main Textual app for Jetbox agent monitoring.

    Layout:
    ┌─────────────────────────────────────┐
    │ Header                              │
    ├─────────────────────────────────────┤
    │ Status Panel (5 lines)              │
    ├─────────────────────────────────────┤
    │ Log Panel (scrollable, fills space) │
    ├─────────────────────────────────────┤
    │ Footer (keybindings)                │
    └─────────────────────────────────────┘
    """

    CSS = """
    #status {
        height: 7;
        background: $boost;
        border: solid $primary;
        padding: 1;
    }

    #logs {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }

    .status-line {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("p", "pause", "Pause", show=True),
        Binding("r", "resume", "Resume", show=True),
        Binding("s", "step", "Step", show=True),
        Binding("c", "context", "Context", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # Reactive state
    paused = reactive(False)
    current_round = reactive(0)
    status_text = reactive("Initializing...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = time.time()
        self.goal = ""
        self.agent_name = ""
        self.model = ""
        self.max_rounds = 50
        self.tokens_used = 0
        self.tokens_max = 128000

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header(show_clock=True)
        yield Static("", id="status")
        yield Log(id="logs", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "Jetbox Agent Monitor"
        self.sub_title = "Press ? for help"
        self._update_status_display()

    def _update_status_display(self) -> None:
        """Update the status panel with current info."""
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        # Calculate progress
        progress_pct = (self.current_round / self.max_rounds * 100) if self.max_rounds > 0 else 0
        progress_bar_width = 30
        filled = int(progress_bar_width * progress_pct / 100)
        progress_bar = "█" * filled + "░" * (progress_bar_width - filled)

        # Token usage
        token_pct = (self.tokens_used / self.tokens_max * 100) if self.tokens_max > 0 else 0

        # Build status text
        status_lines = [
            f"🎯 Goal: {self.goal[:60]}..." if len(self.goal) > 60 else f"🎯 Goal: {self.goal}",
            f"🤖 Agent: {self.agent_name} | Model: {self.model}",
            f"⏱️  Round: {self.current_round}/{self.max_rounds} | Time: {mins}m{secs:02d}s | Status: {self.status_text}",
            f"[{progress_bar}] {progress_pct:.0f}%",
            f"📊 Tokens: {self.tokens_used:,}/{self.tokens_max:,} ({token_pct:.0f}%)",
        ]

        if self.paused:
            status_lines.append("\n⏸️  PAUSED - Press 'r' to resume, 's' to step")

        status_widget = self.query_one("#status", Static)
        status_widget.update("\n".join(status_lines))

    def watch_current_round(self, new_value: int) -> None:
        """Called when current_round changes."""
        self._update_status_display()

    def watch_status_text(self, new_value: str) -> None:
        """Called when status_text changes."""
        self._update_status_display()

    def watch_paused(self, new_value: bool) -> None:
        """Called when pause state changes."""
        self._update_status_display()

    def write_log(self, message: str, style: str = "") -> None:
        """Write a line to the log panel."""
        log_widget = self.query_one("#logs", Log)
        log_widget.write_line(message)

    def action_pause(self) -> None:
        """Pause agent execution."""
        if not self.paused:
            self.paused = True
            self.notify("Agent paused")

    def action_resume(self) -> None:
        """Resume agent execution."""
        if self.paused:
            self.paused = False
            self.notify("Agent resumed")

    def action_step(self) -> None:
        """Execute one step then pause."""
        if self.paused:
            self.notify("Stepping one round...")
            # The display will handle this by checking paused state

    def action_context(self) -> None:
        """Show context inspector (TODO)."""
        self.notify("Context inspector not yet implemented")

    def action_quit(self) -> None:
        """Quit the app."""
        self.exit()


class TextualDisplay(DisplayInterface):
    """
    Textual-based interactive TUI display.

    This wraps the Textual app and provides the DisplayInterface contract.
    """

    def __init__(self):
        """Initialize Textual display."""
        self.app = None
        self._running = False
        self._paused = False
        self._step_requested = False

    def start(self) -> None:
        """Start the Textual app (in background)."""
        self.app = JetboxDashboard()
        self._running = True

        # We need to run the app in a way that doesn't block
        # For now, we'll use a simpler approach - just instantiate it
        # The actual run() will be called by run_agent_with_display()

    def stop(self) -> None:
        """Stop the Textual app."""
        if self.app:
            self.app.exit()
        self._running = False

    def update_status(
        self,
        goal: str,
        agent_name: str,
        model: str,
        current_round: int,
        max_rounds: int,
        elapsed_time: float,
        status: str,
        tokens_used: int | None = None,
        tokens_max: int | None = None,
    ) -> None:
        """Update the status panel."""
        if not self.app:
            return

        self.app.goal = goal
        self.app.agent_name = agent_name
        self.app.model = model
        self.app.current_round = current_round
        self.app.max_rounds = max_rounds
        self.app.status_text = status

        if tokens_used is not None:
            self.app.tokens_used = tokens_used
        if tokens_max is not None:
            self.app.tokens_max = tokens_max

        self.app._update_status_display()

    def log_event(self, event: AgentEvent) -> None:
        """Log an event to the log panel."""
        if not self.app:
            return

        # Icon mapping
        icons = {
            EventType.INFO: "ℹ️",
            EventType.TOOL_CALL: "🔧",
            EventType.TOOL_RESULT: "→",
            EventType.SUCCESS: "✅",
            EventType.WARNING: "⚠️",
            EventType.ERROR: "❌",
            EventType.MILESTONE: "🎉",
            EventType.STATUS_UPDATE: "📊",
        }

        icon = icons.get(event.type, "•")
        timestamp = event.timestamp or datetime.now().strftime("%H:%M:%S")

        message = f"[{timestamp}] {icon} {event.message}"
        self.app.write_log(message)

        # Log details if present
        if event.details:
            for key, value in event.details.items():
                self.app.write_log(f"           {key}: {value}")

    def show_completion(
        self,
        success: bool,
        summary: str,
        duration: float,
        files_created: list[str],
    ) -> None:
        """Show completion screen."""
        if not self.app:
            return

        mins = int(duration // 60)
        secs = int(duration % 60)

        if success:
            self.app.write_log("\n" + "="*50)
            self.app.write_log("🎉 TASK COMPLETED SUCCESSFULLY")
        else:
            self.app.write_log("\n" + "="*50)
            self.app.write_log("❌ TASK FAILED")

        self.app.write_log(f"Duration: {mins}m {secs}s")
        self.app.write_log(f"\n{summary}")

        if files_created:
            self.app.write_log(f"\nFiles created ({len(files_created)}):")
            for f in files_created[:10]:
                self.app.write_log(f"  - {f}")
            if len(files_created) > 10:
                self.app.write_log(f"  ... and {len(files_created) - 10} more")

        self.app.write_log("="*50)

    def prompt_user(self, question: str) -> str:
        """
        Prompt user for input.

        Note: This is tricky in Textual - we may need to implement
        a modal dialog or input widget. For now, fallback to console.
        """
        # TODO: Implement proper Textual input dialog
        return input(f"\n{question}: ")

    def can_pause(self) -> bool:
        """Textual display supports pause/resume."""
        return True

    def is_paused(self) -> bool:
        """Check if currently paused."""
        if self.app:
            return self.app.paused
        return False

    def wait_if_paused(self) -> None:
        """Block until user resumes."""
        if not self.app:
            return

        while self.app.paused:
            time.sleep(0.1)
            # TODO: Proper async event handling
