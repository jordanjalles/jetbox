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
        self.status_bar_lines = 8  # Number of lines reserved for status bar (1 header + 1 round + 1 context + 4 latency + 1 status)

    def start(self) -> None:
        """Enter alternate screen buffer and redirect stdout for TUI mode (unless in verbose mode)."""
        import sys
        import os

        # Skip alternate screen mode in non-verbose mode (chatbot/interactive mode)
        if not self.verbose:
            return

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
            # Reserve self.status_bar_lines at bottom for multi-line status
            scroll_bottom = self.terminal_height - self.status_bar_lines
            sys.stdout.write(f'\033[1;{scroll_bottom}r')
            sys.stdout.flush()

            # Position cursor at line above status bar so logs appear attached to progress line
            # This prevents the large empty space between top and bottom
            sys.stdout.write(f'\033[{scroll_bottom};1H')
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

    def _format_latency_histogram(self, latency_history: list[float], width: int = 20, height: int = 4) -> list[str]:
        """
        Format latency history as a multi-line vertical histogram.

        Args:
            latency_history: List of recent round latencies in seconds
            width: Width of histogram in characters (number of bars)
            height: Height of histogram in lines

        Returns:
            List of strings representing each line of the histogram (top to bottom)
        """
        if not latency_history:
            return ["░" * width] * height

        # Take last N entries to fit width
        recent = latency_history[-width:]

        # Normalize to 0-height range
        min_lat = min(recent)
        max_lat = max(recent)
        range_lat = max_lat - min_lat if max_lat > min_lat else 1.0

        # Calculate bar heights (0 to height)
        bar_heights = []
        for lat in recent:
            normalized = (lat - min_lat) / range_lat
            # Map to 0-height range
            bar_height = normalized * height
            bar_heights.append(bar_height)

        # Build histogram lines from top to bottom
        lines = []
        for row in range(height):
            line = ""
            # Row threshold: top row (row=0) needs height >= 4.0, bottom row (row=3) needs height >= 1.0
            threshold = height - row

            for bar_height in bar_heights:
                if bar_height >= threshold:
                    # Full block for this row
                    line += "█"
                elif bar_height >= threshold - 1:
                    # Partial block - use fractional blocks
                    fraction = bar_height - (threshold - 1)
                    if fraction >= 0.75:
                        line += "▇"
                    elif fraction >= 0.5:
                        line += "▆"
                    elif fraction >= 0.25:
                        line += "▃"
                    else:
                        line += "▁"
                else:
                    # Empty space
                    line += " "

            # Pad if shorter than width
            line += " " * (width - len(line))
            lines.append(line)

        return lines

    def _format_size(self, chars: int) -> str:
        """Format character count as human-readable size (e.g., 12K, 1.5M)."""
        if chars < 1000:
            return f"{chars}"
        elif chars < 1000000:
            return f"{chars // 1000}K"
        else:
            return f"{chars / 1000000:.1f}M"

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
        context_breakdown: dict[str, int] | None = None,
        latency_history: list[float] | None = None,
        detailed_status: str | None = None,
        spinner_frame: str | None = None,
    ) -> None:
        """Update multi-line status display at bottom of screen."""
        # Calculate progress percentage
        progress = int((current_round / max_rounds) * 100)
        mins = int(elapsed_time // 60)
        secs = int(elapsed_time % 60)

        if not self.verbose:
            # Minimal mode (chatbot): simple single-line status
            # Show model, context size, and current activity
            if context_breakdown:
                total = sum(context_breakdown.values())
                total_size = self._format_size(total)
                ctx_info = f"Context:{total_size}"
            else:
                ctx_info = "Context:0"

            # Format detailed status nicely
            status_emoji = {
                "starting": "🚀",
                "building_context": "🔧",
                "calling_llm": "💭",
                "processing_response": "⚙️",
                "writing": "✍️",
                "reading": "📖",
                "testing": "🧪",
                "executing": "⚡",
                "completing": "✅",
                "ready": "✓",
                "error": "❌",
            }
            emoji = status_emoji.get(detailed_status or "", "▶️")
            status_text = (detailed_status or status).replace("_", " ").capitalize()

            # Add spinner if in LLM call
            if spinner_frame:
                status_display = f"{emoji} {status_text} {spinner_frame}"
            else:
                status_display = f"{emoji} {status_text}"

            status_line = f"[{agent_name}] {ctx_info} | {status_display}"
            lines = [status_line]
        else:
            # Verbose mode: 8-line status display

            # Line 1: Agent name header (bordered style)
            agent_display_name = agent_name.replace('_', ' ').title()
            header_width = 70
            equals_count = (header_width - len(agent_display_name) - 2) // 2
            header_line = "=" * equals_count + f" {agent_display_name} " + "=" * equals_count
            # Ensure exact width
            if len(header_line) < header_width:
                header_line += "=" * (header_width - len(header_line))

            # Line 2: Round allowance progress bar
            bar_width = 20
            filled = int((progress / 100) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            line2 = f"[Round Allowance] [{bar}] {progress:3d}% ({current_round}/{max_rounds}) | ⏱️  {mins}m{secs:02d}s"

            # Line 3: Context composition breakdown
            if context_breakdown:
                sys_size = self._format_size(context_breakdown.get("system", 0))
                user_size = self._format_size(context_breakdown.get("user", 0))
                asst_size = self._format_size(context_breakdown.get("assistant", 0))
                tool_size = self._format_size(context_breakdown.get("tool", 0))
                total = sum(context_breakdown.values())
                total_size = self._format_size(total)

                # Estimate max context (128K tokens = ~512K chars)
                max_context = 512000
                ctx_pct = int((total / max_context) * 100) if max_context else 0

                line3 = f"[Context]  Sys:{sys_size}  User:{user_size}  Asst:{asst_size}  Tool:{tool_size}  Total:{total_size} ({ctx_pct}%)"
            else:
                line3 = "[Context]  (no data)"

            # Lines 4-7: Latency histogram (4 lines tall)
            lines = [header_line, line2, line3]

            if latency_history:
                histogram_lines = self._format_latency_histogram(latency_history, width=20, height=4)
                avg = sum(latency_history) / len(latency_history)
                last = latency_history[-1] if latency_history else 0
                min_lat = min(latency_history)
                max_lat = max(latency_history)

                # Add histogram with label on first line
                lines.append(f"[Latency]  {histogram_lines[0]}")
                for hist_line in histogram_lines[1:]:
                    lines.append(f"           {hist_line}")

                # Add statistics line after histogram
                stats_line = f"           Avg:{avg:.1f}s Last:{last:.1f}s Min:{min_lat:.1f}s Max:{max_lat:.1f}s"
                lines.append(stats_line)
            else:
                lines.append("[Latency]  (no data)")
                lines.extend([""] * 3)  # Placeholder lines

            # Line 8: Granular status
            status_emoji = {
                "starting": "🚀",
                "building_context": "🔧",
                "calling_llm": "💭",
                "processing_response": "⚙️",
                "writing": "✍️",
                "reading": "📖",
                "testing": "🧪",
                "executing": "⚡",
                "completing": "✅",
                "error": "❌",
            }
            emoji = status_emoji.get(detailed_status or "", "▶️")
            # Format status: replace underscores with spaces and capitalize
            status_text = (detailed_status or status).replace("_", " ").capitalize()

            # Add spinner if provided (only during "calling_llm" state)
            if spinner_frame:
                status_line = f"[Status]  {emoji} {status_text} {spinner_frame}"
            else:
                status_line = f"[Status]  {emoji} {status_text}"

            lines.append(status_line)

        # Update status at bottom rows using absolute positioning
        if self.using_alt_screen and self.original_stdout:
            # Save cursor, move to status area, clear and write lines, restore cursor
            output = "\0337"  # Save cursor

            for i, line in enumerate(lines):
                row = self.terminal_height - (self.status_bar_lines - i - 1)
                output += f"\033[{row};1H\033[K{line}"

            output += "\0338"  # Restore cursor

            self.original_stdout.write(output)
            self.original_stdout.flush()
        else:
            # Not in alt screen - just print normally
            import sys
            for line in lines:
                sys.stdout.write(line + '\n')
            sys.stdout.flush()

        self.last_status_line = lines[0] if lines else ""

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
