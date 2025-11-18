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
        self.using_alt_screen = False  # Track if we're using alternate screen
        self.log_row = 1  # Start logs at row 1 (status will be at bottom)
        self.original_stdout = None  # Store original stdout for restoration
        self.terminal_height = 24  # Default terminal height (will be detected)

    def start(self) -> None:
        """Enter alternate screen buffer and redirect stdout for TUI mode."""
        import sys
        import os
        if sys.stdout.isatty():
            # Get terminal height
            try:
                size = os.get_terminal_size()
                self.terminal_height = size.lines
            except OSError:
                self.terminal_height = 24  # Fallback

            # Enter alternate screen buffer and clear it
            sys.stdout.write('\033[?1049h\033[2J')
            sys.stdout.flush()
            self.using_alt_screen = True

            # Set scrolling region to prevent logs from overwriting status bar
            # Format: \033[<top>;<bottom>r sets scroll region
            # We want rows 1 to (height-1) to scroll, keeping last row for status
            sys.stdout.write(f'\033[1;{self.terminal_height - 1}r')
            sys.stdout.flush()

            # Save original stdout and replace with our wrapper
            self.original_stdout = sys.stdout
            sys.stdout = self._TUIStdout(self)

    def stop(self) -> None:
        """Restore stdout and exit alternate screen buffer."""
        import sys
        if self.using_alt_screen:
            # Restore original stdout
            if self.original_stdout:
                sys.stdout = self.original_stdout
                self.original_stdout = None

            # Exit alternate screen buffer
            sys.stdout.write('\033[?1049l')
            sys.stdout.flush()
            self.using_alt_screen = False

    class _TUIStdout:
        """Wrapper for stdout that positions output in TUI."""
        def __init__(self, display):
            self.display = display
            self._buffer = ""

        def write(self, text):
            """Intercept writes and position them in TUI."""
            if not text:
                return

            # Add to buffer
            self._buffer += text

            # If we have complete lines, print them
            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                if line:  # Don't print empty lines
                    # Just write the line - scrolling region handles positioning
                    # Logs scroll naturally, status bar stays pinned at bottom
                    self.display.original_stdout.write(line + '\n')
                    self.display.original_stdout.flush()

        def flush(self):
            """Flush buffer."""
            if self._buffer:
                self.display.original_stdout.write(self._buffer)
                self.display.original_stdout.flush()
                self._buffer = ""
            self.display.original_stdout.flush()

        def __getattr__(self, name):
            """Delegate other attributes to original stdout."""
            return getattr(self.display.original_stdout, name)

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
        """Update status line at fixed position (BOTTOM row)."""
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

        # Update status at BOTTOM row using absolute positioning
        # Write directly to original stdout (bypass wrapper)
        if self.using_alt_screen and self.original_stdout:
            # Save cursor position, move to bottom row, clear line, write status, restore cursor
            # This way logs keep scrolling while status stays at bottom
            self.original_stdout.write(f'\0337\033[{self.terminal_height};1H\033[K{status_line}\0338')
            self.original_stdout.flush()
        else:
            # Not in alt screen - just print normally to current stdout
            import sys
            sys.stdout.write(status_line + '\n')
            sys.stdout.flush()

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
