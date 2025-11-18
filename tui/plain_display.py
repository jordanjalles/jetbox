"""
Plain Display - Traditional print() output.

This is the fallback display for:
- Non-TTY environments (pipes, redirects)
- CI/CD systems
- Users who prefer simple output
- Emergency rollback if TUI has issues
"""

from .display_interface import DisplayInterface, AgentEvent, EventType


class PlainDisplay(DisplayInterface):
    """
    Simple print()-based display.

    This is essentially the current Jetbox behavior - just cleaner.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize plain display.

        Args:
            verbose: If True, show all events. If False, show minimal output.
        """
        self.verbose = verbose
        self.last_status_line = ""  # Track last status for inline updates

    def start(self) -> None:
        """No initialization needed for plain display."""
        pass

    def stop(self) -> None:
        """No cleanup needed for plain display."""
        pass

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
        """Update status line IN PLACE (overwrites previous status)."""
        # Calculate progress percentage
        progress = int((current_round / max_rounds) * 100)
        mins = int(elapsed_time // 60)
        secs = int(elapsed_time % 60)

        if not self.verbose:
            # Minimal mode: just show progress
            status_line = f"[{progress:3d}%] Round {current_round}/{max_rounds} - {status}"
        else:
            # Verbose mode: full status line with progress bar
            # Create progress bar (20 chars wide)
            bar_width = 20
            filled = int((progress / 100) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            status_line = f"📊 [{bar}] {progress:3d}% | Round {current_round}/{max_rounds} | ⏱️  {mins}m{secs:02d}s | {status}"

            if tokens_used and tokens_max:
                pct = int((tokens_used / tokens_max) * 100)
                status_line += f" | 🧠 {tokens_used}/{tokens_max} ({pct}%)"

        # Clear previous line and print new status (IN PLACE update)
        # Use \r to return to start of line, then overwrite with spaces to clear old content
        clear_line = "\r" + " " * len(self.last_status_line) + "\r"
        print(clear_line + status_line, end="", flush=True)
        self.last_status_line = status_line

    def log_event(self, event: AgentEvent) -> None:
        """Print event to stdout."""
        if not self.verbose and event.type not in [EventType.ERROR, EventType.MILESTONE]:
            # In minimal mode, only show errors and milestones
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
        timestamp = event.timestamp or ""

        if timestamp:
            print(f"{timestamp} {icon} {event.message}")
        else:
            print(f"{icon} {event.message}")

        # Print details if present and in verbose mode
        if self.verbose and event.details:
            for key, value in event.details.items():
                print(f"   {key}: {value}")

    def show_completion(
        self,
        success: bool,
        summary: str,
        duration: float,
        files_created: list[str],
    ) -> None:
        """Print completion summary."""
        print("\n" + "="*70)
        if success:
            print("✅ TASK COMPLETED SUCCESSFULLY")
        else:
            print("❌ TASK FAILED")
        print("="*70)

        mins = int(duration // 60)
        secs = int(duration % 60)
        print(f"Duration: {mins}m {secs}s")
        print(f"\n{summary}")

        if files_created:
            print(f"\nFiles created ({len(files_created)}):")
            for f in files_created[:10]:  # Show first 10
                print(f"  - {f}")
            if len(files_created) > 10:
                print(f"  ... and {len(files_created) - 10} more")

        print("="*70 + "\n")

    def prompt_user(self, question: str) -> str:
        """Simple input() prompt."""
        return input(f"{question}: ")
